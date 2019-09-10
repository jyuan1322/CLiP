import os.path
import numpy as np
import sys
from scipy import integrate
from scipy.stats import norm
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pprint import pprint
from CLiPY_utils import generate_pss_model_simple, \
                  calc_var_from_geno, \
                  generate_population




def heterogeneity(cases,controls,clist,snp_props):
    """
    cases, controls: Numpy array where each row is an indiv and each col is a snp
    clist: tuple of indices that are the snps for DB
    """
    num_snps = len(clist)
    snp_cases_all = cases[:,clist]
    snp_controls_all = controls[:,clist]

    snp_cases = snp_cases_all
    snp_controls = snp_controls_all

    num_cases = snp_cases_all.shape[0]
    num_controls = snp_controls_all.shape[0]

    N = float(len(snp_cases))
    Np = float(len(snp_controls))
    R = np.corrcoef(snp_cases.T)
    Rp = np.corrcoef(snp_controls.T)
    print("N:", N)
    print("Np:", Np)
    # NOTE: (Np-N) not (Np+N) to account for controls including all individuals
    Y = np.sqrt(N*Np/(Np-N)) * (R-Rp)
    print("Y:", Y)

    pi_cases = np.sum(snp_cases, axis=0) / (2*snp_cases.shape[0])
    pi_controls = np.sum(snp_controls, axis=0) / (2*snp_controls.shape[0])
    gamma = pi_cases/(1-pi_cases) / (pi_controls/(1-pi_controls))

    # calculate heterogeneity score
    elem1 = np.sqrt(pi_controls*(1-pi_controls))
    elem2 = gamma-1
    elem3 = elem2 * pi_controls + 1
    mat1 = np.sqrt(np.dot(elem1.reshape((num_snps, 1)), elem1.reshape((1, num_snps))))
    mat2 = np.dot(elem2.reshape((num_snps, 1)), elem2.reshape((1, num_snps)))
    mat3 = np.dot(elem3.reshape((num_snps, 1)), elem3.reshape((1, num_snps)))
    w = mat1 * mat2 / mat3
    score = np.sum(np.triu(w*Y, k=1)) / np.sqrt(np.sum(np.triu(w ** 2, k=1)))
    return score

def log1(phenos, mean, std):
    percentiles = norm.cdf(phenos, loc=mean, scale=std)
    return -(np.log(1 - percentiles))
def log1p5(phenos, mean, std):
    percentiles = norm.cdf(phenos, loc=mean, scale=std)
    return (-np.log(1 - percentiles))**1.5
def log3(phenos, mean, std):
    percentiles = norm.cdf(phenos, loc=mean, scale=std)
    return -(np.log(1 - percentiles))**3

def step(phenos, mean, std):
    percentiles = norm.cdf(phenos, loc=mean, scale=std)
    return (percentiles > 0.5)

def polynom2(phenos, mean, std):
    coefs = [-409.52448861055984,431.6513313140901,38.00764396532995]

    percentiles = norm.cdf(phenos, loc=mean, scale=std)
    polynom = np.poly1d(coefs)
    return np.maximum(0, polynom(percentiles))

def polynom4(phenos, mean, std):
    coefs = [-540.0797504985338,-360.26557093065634,-44.21132533248787,732.5017720012906,234.54891527783096]

    percentiles = norm.cdf(phenos, loc=mean, scale=std)
    polynom = np.poly1d(coefs)
    return np.maximum(0, polynom(percentiles))

def polynom6(phenos, mean, std):
    coefs = [-272.72479315758346,484.8418136747146,-270.8586221944605,-185.17269248431845,-426.3074761490136,582.3584611990045,88.95092357490671]

    percentiles = norm.cdf(phenos, loc=mean, scale=std)
    polynom = np.poly1d(coefs)
    return np.maximum(0, polynom(percentiles))

def sigmoid(phenos, mean, std, scale=20, pc_center=0.5):
    percentiles = norm.cdf(phenos, loc=mean, scale=std)
    return 1.0/(1.0 + np.exp(-scale*(percentiles-pc_center)))

def linear(phenos, mean, std):
    percentiles = norm.cdf(phenos, loc=mean, scale=std)
    return percentiles

def get_weights(phenos, mean, std, weight_func,
                 phi_marg=None, marg_only_num_inds=None):
    if marg_only_num_inds is not None: # return the marginalization factor only
        lowery = mean - 4*std
        uppery = mean + 4*std
        x = np.linspace(lowery, uppery, marg_only_num_inds)
        #x = np.random.normal(loc=mean, scale=std, size=marg_only_num_inds)
        y = [get_weights(y, mean, std, weight_func) for y in x]
        return np.sum(y)

    weights = weight_func(phenos, mean, std)
    if phi_marg is None:
        return weights
    else:
        return weights / phi_marg

def corr(x, w):
    """Weighted Correlation"""
    c = np.cov(x, aweights=w)
    d = np.diag(np.diag(c) ** -0.5)
    return np.dot(np.dot(d, c), d)


def expected_continuous_heterogeneity(rho, clist, snp_props, exp_phi, prs_mean, prs_sigma, num_inds, weight_func):
    """
    cases, controls: Numpy array where each row is an indiv and each col is a snp
    clist: tuple of indices that are the snps for DB
    snp_props:
    """
    ps, betas = snp_props
    num_snps = len(clist)

    pi_plus = exp_phi / 2
    pi_minus = ps

    print("expected pi_plus, pi_minus")
    print(pi_plus)
    print(pi_minus)

    gamma = pi_plus/(1-pi_plus) / (pi_minus/(1-pi_minus))
    print("expected gamma:", gamma)

    n = float(num_snps)
    N = num_inds
    lowery = prs_mean - 10*prs_sigma
    uppery = prs_mean + 10*prs_sigma
    x = np.random.normal(loc=prs_mean, scale=prs_sigma, size=num_inds)
    y = np.array([get_weights(y, prs_mean, prs_sigma, weight_func=weight_func) for y in x])
    y = y / np.sum(y)
    w2 = np.sum(np.square(y))
    print("expected w2:", w2)

    R = rho
    Rp = np.identity(R.shape[0])
    Y = (w2 - 1/N)**-0.5 * (R-Rp)

    # calculate heterogeneity score
    elem1 = np.sqrt(pi_minus*(1-pi_minus))
    elem2 = gamma-1
    elem3 = elem2 * pi_minus + 1
    mat1 = np.sqrt(np.dot(elem1.reshape((num_snps, 1)), elem1.reshape((1, num_snps))))
    mat2 = np.dot(elem2.reshape((num_snps, 1)), elem2.reshape((1, num_snps)))
    mat3 = np.dot(elem3.reshape((num_snps, 1)), elem3.reshape((1, num_snps)))
    w = mat1 * mat2 / mat3
    score = np.sum(np.triu(w*Y, k=1)) / np.sqrt(np.sum(np.triu(w ** 2, k=1)))
    return score


def get_mats_from_pop(pop, phen, z_thresh):
    genos, phenos = pop
    mu = np.mean(phenos)
    sigma = np.std(phenos)

    cases_indices = np.where(phenos > (mu + z_thresh * sigma))
    cases = genos[cases_indices]
    return cases, genos # return full pop as controls

def run_heterogeneity_on_pop(pop, snps, snp_phens=0, case_phen=1, z=1.5):
    cases, controls = get_mats_from_pop(pop, case_phen, z)

    clist = range(len(snps[0]))
    return heterogeneity(cases, controls, clist, snps)

def run_cont_heterogeneity_on_pop(pop, snps, weight_func, snp_phens=0, case_phen=0, **kwargs):
    genos, phenos = pop
    clist = range(len(snps[0]))

    num_snps = len(clist)
    snp_indivs = genos[:,clist]
    num_indivs = snp_indivs.shape[0]

    symmetric = False
    if "symmetric" in kwargs:
        symmetric = kwargs["symmetric"]

    if "coef_wts" in kwargs:
        # testing optimal weight function (polynomial coefficients)
        if symmetric:
            percentiles = norm.cdf(phenos, loc=np.mean(phenos), scale=np.std(phenos)) - 0.5
        else:
            percentiles = norm.cdf(phenos, loc=np.mean(phenos), scale=np.std(phenos))
        coef_wts = kwargs["coef_wts"]
        polynom = np.poly1d(coef_wts,r=False)
        weights = [polynom(x) for x in percentiles]
        weights = weights / np.sum(weights)

    elif "block_wts" in kwargs and "numbins" in kwargs:
        # for testing of optimal weight function
        percentiles = norm.cdf(phenos, loc=np.mean(phenos), scale=np.std(phenos))
        block_wts = kwargs["block_wts"]
        numbins = kwargs["numbins"]
        weights = block_wts[[int(x) for x in np.floor(percentiles * numbins)]]
        weights = weights / np.sum(weights)
    else:
        # normal application of weight function
        weights = get_weights(phenos, np.mean(phenos), np.std(phenos), weight_func=weight_func)
        weights = weights / np.sum(weights)

    pi_plus = np.sum(snp_indivs * weights.reshape((num_indivs, 1)), axis=0) / 2
    pi_minus = np.sum(snp_indivs, axis=0) / (2*float(num_indivs))

    gamma = pi_plus/(1-pi_plus) / (pi_minus/(1-pi_minus))

    n = float(num_snps)
    N = num_indivs
    w2 = np.sum(weights ** 2)
    try:
        R = corr(snp_indivs.T, weights)
    except:
        print(coef_wts)
        sys.exit(0)

    # subtracting the reverse weights
    Rp = np.corrcoef(snp_indivs.T)

    Y = (w2 - 1/N)**-0.5 * (R-Rp) # verify the sign of the scaling factor

    # calculate heterogeneity score
    elem1 = np.sqrt(pi_minus*(1-pi_minus))
    elem2 = gamma-1
    elem3 = elem2 * pi_minus + 1
    mat1 = np.sqrt(np.dot(elem1.reshape((num_snps, 1)), elem1.reshape((1, num_snps))))
    mat2 = np.dot(elem2.reshape((num_snps, 1)), elem2.reshape((1, num_snps)))
    mat3 = np.dot(elem3.reshape((num_snps, 1)), elem3.reshape((1, num_snps)))
    w = mat1 * mat2 / mat3
    score = np.sum(np.triu(w*Y, k=1)) / np.sqrt(np.sum(np.triu(w ** 2, k=1)))
    return score

def get_ex_y(beta, p, y, mean, sigma):
    """
        sigma: the standard deviation of the PRS distribution
    """
    xvals = np.array([0,1,2])
    py_x = np.empty(3)
    px = np.array([(1-p)**2, 2*p*(1-p), p**2])
    for i in range(3):
        py_x[i] = norm.pdf(y, loc=mean -2*p*beta + xvals[i]*beta,
                              scale=np.sqrt(sigma**2 - 2*p*(1-p)*beta**2))
    probs = np.multiply(xvals, np.multiply(py_x, px))
    marg = np.sum(np.multiply(py_x,px))
    return np.sum(probs)/marg

def get_ex2_y(beta, p, y, mean, sigma):
    """
        sigma: the standard deviation of the PRS distribution
    """
    xvals = np.array([0,1,2])
    py_x = np.empty(3)
    px = np.array([(1-p)**2, 2*p*(1-p), p**2])
    for i in range(3):
        py_x[i] = norm.pdf(y, loc=mean - 2*p*beta + xvals[i]*beta,
                              scale=np.sqrt(sigma**2 - 2*p*(1-p)*beta**2))
    probs = np.multiply(np.square(xvals), np.multiply(py_x, px))
    marg = np.sum(np.multiply(py_x,px))
    return np.sum(probs)/marg

def get_exx_y(beta1, beta2, p1, p2, y, mean, sigma):
    """
        sigma: the standard deviation of the PRS distribution
    """
    xvals1 = np.array([0,1,2])
    xvals2 = np.array([0,1,2])
    py_xx = np.empty((3,3))
    px1 = np.array([p1**2, 2*p1*(1-p1), (1-p1)**2])
    px2 = np.array([p2**2, 2*p2*(1-p2), (1-p2)**2])
    for i in range(3):
        for j in range(3):
            py_xx[i,j] = norm.pdf(y, loc=mean - 2*p1*beta1 - 2*p2*beta2 \
                                              + xvals1[i]*beta1 + xvals2[j]*beta2,
                                     scale=np.sqrt(sigma**2 - 2*p1*(1-p1)*beta1**2 \
                                                            - 2*p2*(1-p2)*beta2**2))

    probs = np.multiply(np.outer(xvals1,xvals2), np.multiply(py_xx, np.outer(px1,px2)))
    marg = np.sum(np.multiply(py_xx, np.outer(px1,px2)))
    return np.sum(probs)/marg

def get_exp_heterogeneity(num_inds, snps, h, weight_func):
    ps, betas = snps

    # calculate distribution of PRSs
    beta_var = calc_var_from_geno(snps)
    prs_sigma = np.sqrt( beta_var + (1-h)/h * beta_var)
    prs_mean = 2 * np.sum(np.multiply(betas, ps))

    # get expected correlation - input this into expected het score calculation
    num_snps = len(ps)
    ex_phi = np.empty(num_snps)
    ex2_phi = np.empty(num_snps)
    exx_phi = np.zeros((num_snps, num_snps))

    phi_marg = get_weights(None, prs_mean, prs_sigma, weight_func,
                                 phi_marg=None, marg_only_num_inds=num_inds)
    for i in range(num_snps):
        # calculate E[X|y] where y is the PRS --> get_ex_y

        lowery = prs_mean - 4*prs_sigma
        uppery = prs_mean + 4*prs_sigma

        marg = integrate.quad(lambda y: get_weights(phenos=y, mean=prs_mean, std=prs_sigma,
                                                    weight_func=weight_func, phi_marg=phi_marg) * \
                                        norm.pdf(y, loc=prs_mean, scale=prs_sigma),
                              lowery, uppery)[0]
        ex_phi[i] = 1/marg * integrate.quad(lambda y: get_weights(phenos=y, mean=prs_mean, std=prs_sigma,
                                                                  weight_func=weight_func, phi_marg=phi_marg) * \
                                                      get_ex_y(betas[i], ps[i], y, prs_mean, prs_sigma) * \
                                                      norm.pdf(y, loc=prs_mean, scale=prs_sigma),
                                            lowery, uppery)[0]
        ex2_phi[i] = 1/marg * integrate.quad(lambda y: get_weights(phenos=y, mean=prs_mean, std=prs_sigma,
                                                                   weight_func=weight_func, phi_marg=phi_marg) * \
                                                       get_ex2_y(betas[i], ps[i], y, prs_mean, prs_sigma) * \
                                                       norm.pdf(y, loc=prs_mean, scale=prs_sigma),
                                            lowery, uppery)[0]
        for j in range(i+1, num_snps):
            exx_phi[i,j] = 1/marg * integrate.quad(lambda y: get_weights(phenos=y, mean=prs_mean, std=prs_sigma,
                                                                         weight_func=weight_func, phi_marg=phi_marg) * \
                                                             get_exx_y(betas[i], betas[j],
                                                                       ps[i], ps[j], y,
                                                                       prs_mean, prs_sigma) * \
                                                             norm.pdf(y, loc=prs_mean, scale=prs_sigma),
                                                   lowery, uppery)[0]

    # calculate expected correlation
    rho = np.empty((num_snps,num_snps))
    for i in range(num_snps):
        rho[i,i] = 1.0

        for j in range(i+1,num_snps):
            rho[i,j] = (exx_phi[i,j] - ex_phi[i] * ex_phi[j]) / \
                        np.sqrt(ex2_phi[i] - ex_phi[i]**2) / np.sqrt(ex2_phi[j] - ex_phi[j]**2)
            rho[j,i] = rho[i,j]

    clist = range(num_snps)
    return expected_continuous_heterogeneity(rho, clist, snps, ex_phi, prs_mean, prs_sigma, num_inds, weight_func)


