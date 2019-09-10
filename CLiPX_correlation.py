import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle, os, sys, argparse
from collections import OrderedDict
from scipy import integrate
from scipy.stats import norm, multivariate_normal



def probit(x, effects, thresh, cont_PRS_sd, hLsq):
    #res = 1.0 - norm.cdf((np.dot(x, effects) - thresh) / np.sqrt(1-hLsq))
    res = norm.cdf((np.dot(x, effects) - thresh) / np.sqrt(1-hLsq))
    return res

"""
When X are correlated, need adjustment to variance of remaining X_{-i}
"""
def probit_corr(x, xind, effects, freqs, thresh, hLsq, g2w_ecovs):
    # create partitioned covariance matrices
    idxs = [i for i in range(len(effects)) if i not in xind]
    Sig11 = g2w_ecovs[idxs,:]
    Sig11 = Sig11[:,idxs]
    Sig12 = g2w_ecovs[:,xind]
    Sig12 = Sig12[idxs,:]
    Sig22 = g2w_ecovs[xind,:]
    Sig22 = Sig22[:,xind]


    mu_rem = np.dot(np.dot(Sig12, np.linalg.inv(Sig22)), np.array(x).reshape(-1,1))
    sig_rem = Sig11 - np.dot(np.dot(Sig12, np.linalg.inv(Sig22)), Sig12.T)

    mu_prs = np.dot(x, effects[xind]) + np.dot(mu_rem.flatten(), effects[idxs])
    var_prs = (1-hLsq)
    for i in idxs:
        new_var = effects[i]**2
        for j in idxs:
            if j > i:
                new_var += 2 * g2w_ecovs[i,j] * effects[i] * effects[j]
        var_prs += new_var
    res = norm.cdf((mu_prs - thresh) / np.sqrt(var_prs))

    return res

def get_control_cov(freqs, gen2expr_wgtmat):
    # evaluate correlation between cases and control
    geno_vars = 2.0 * np.multiply(freqs, 1-freqs)
    num_expr = gen2expr_wgtmat.shape[0] # expr x genes
    g2w_ecovs = np.empty((num_expr, num_expr))
    for i in range(num_expr):
        g2w_ecovs[i,i] = 1.0
        for j in range(i+1,num_expr):
            # no off-diagonal covariance between genotypes
            g2w_ecovs[i,j] = np.sum(np.multiply(
                              np.multiply(gen2expr_wgtmat[i,:],
                                          gen2expr_wgtmat[j,:]),geno_vars))
            g2w_ecovs[j,i] = g2w_ecovs[i,j]
    return g2w_ecovs

def expected_corr_unnorm(alpha, thresh, freqs, gen2expr_wgtmat, heritability, verbose=False):
    """
    Return an MxM matrix of expected correlations
    """
    g2w_ecovs = get_control_cov(freqs, gen2expr_wgtmat)
    
    num_expr = gen2expr_wgtmat.shape[0] # expr x genes
    effects = alpha
    # calculate E[X], E[X^2]
    ex = []
    ex2 = []
    for i in range(num_expr):
        dist = 4 # increasing this will increase accuracy but slow execution
        lowerx = -dist
        upperx = dist
        
        # quad returns tuple with integral and upper bound on error
        marg = integrate.quad(lambda x: probit_corr([x], [i], effects, freqs, thresh, heritability, g2w_ecovs) * \
                              norm.pdf(x, loc=0, scale=1),
                              lowerx, upperx)[0]

        ex.append(1.0/marg * integrate.quad(lambda x: x * probit_corr([x], [i], effects, freqs, thresh, heritability, g2w_ecovs) * \
                                            norm.pdf(x, loc=0, scale=1),
                                            lowerx, upperx)[0])

        ex2.append(1.0/marg * integrate.quad(lambda x: x**2 * probit_corr([x], [i], effects, freqs, thresh, heritability, g2w_ecovs) * \
                                             norm.pdf(x, loc=0, scale=1),
                                             lowerx, upperx)[0])

    # calculate E[XY]
    exy = np.empty((num_expr,num_expr))
    for i in range(num_expr):
        for j in range(i+1,num_expr):
            efi = effects[i]
            efj = effects[j]
            print("double integral i:",i,"j:",j)
            dist = 4
            lowerx = -dist
            upperx = dist
            lowery = -dist
            uppery = dist

            # take into account correlation between X and Y
            mvn_mean = [0,0]
            mvn_cov = [[g2w_ecovs[i,i], g2w_ecovs[i,j]],
                       [g2w_ecovs[i,j], g2w_ecovs[j,j]]]

            marg = integrate.dblquad(lambda y,x: probit_corr([x,y], [i,j], effects, freqs, thresh, heritability, g2w_ecovs) * \
                                     multivariate_normal.pdf([x,y],
                                                             mean=mvn_mean,
                                                             cov=mvn_cov),
                                     lowerx, upperx, lowery, uppery)[0]
                                     

            exy[i,j] = 1.0/marg * integrate.dblquad(lambda y,x: x * y * \
                                                    probit_corr([x,y], [i,j], effects, freqs, thresh, heritability, g2w_ecovs) * \
                                                    multivariate_normal.pdf([x,y],
                                                                            mean=mvn_mean,
                                                                            cov=mvn_cov),
                                                    lowerx, upperx, lowery, uppery)[0]

    # calculate expected correlation
    rho = np.empty((num_expr,num_expr))
    for i in range(num_expr):
        rho[i,i] = 1.0
        for j in range(i+1,num_expr):
            rho[i,j] = (exy[i,j] - ex[i] * ex[j]) / np.sqrt(ex2[i] - ex[i]**2) / np.sqrt(ex2[j] - ex[j]**2)
            rho[j,i] = rho[i,j]

    return rho, ex


