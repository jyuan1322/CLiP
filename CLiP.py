import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle, sys, argparse
from scipy.stats import norm


def generate_cohort(num_cases,num_conts,freqs,betas,h_sq,thresh):
    num_snps = len(freqs)
    cases = np.empty((num_cases,num_snps))
    conts = np.empty((num_conts,num_snps))

    # generate controls
    conts = np.random.binomial(n=2, p=freqs, size=(num_conts,num_snps))

    # generate cases
    cur_cases = 0
    subset_size = 10000
    exp_score = 2*np.dot(betas, freqs)
    while cur_cases < num_cases:
        case_samples = np.random.binomial(n=2, p=freqs, size=(subset_size,num_snps))
        scores = np.dot(case_samples, betas) + \
                 np.random.normal(0,np.sqrt(1-h_sq),subset_size) - \
                 exp_score # set mean score to 0
        idxs = np.where(scores > thresh)[0]

        if cur_cases+len(idxs) > num_cases:
            idxs = idxs[:(num_cases-cur_cases)]
        cases[cur_cases:cur_cases + len(idxs),:] = case_samples[idxs,:]
        cur_cases += len(idxs)
        print(cur_cases)
    if cases.shape[0] > num_cases:
        cases = cases[:num_cases, :]
    return cases,conts






def probit_corr(x, effect, freq, thresh):
    var_prs = effect**2 * 2 * freq*(1-freq)
    ex = 2*freq
    res = norm.cdf((effect*(x-ex) - thresh) / np.sqrt(1-var_prs))
    return res

def probit_corr_xy(x,y, efi, efj, freqi, freqj, thresh):
    var_prs = efi**2 * 2 * freqi*(1-freqi) + efj**2 * 2 * freqj*(1-freqj)
    ex = 2*freqi
    ey = 2*freqj
    res = norm.cdf((efi*(x-ex) + efj*(y-ey) - thresh) / np.sqrt(1-var_prs))
    return res

def expected_corr_unnorm(effects, thresh, freqs, heritability, verbose=False):

    num_snps = len(effects)
    # calculate E[X], E[X^2]
    ex = []
    ex2 = []
    for i in range(num_snps):
        margx = []
        for x in range(3):
            m = probit_corr(x, effects[i], freqs[i], thresh) * freqs[i]**x * (1-freqs[i])**(1-x)
            if x==1:
                m *= 2
            margx.append(m)
        margx = np.array(margx) / np.sum(margx)
        ex.append(margx[1]*1 + margx[2]*2)
        ex2.append(margx[1]*1 + margx[2]*4)

    # calculate E[XY]
    exy = np.empty((num_snps,num_snps))
    for i in range(num_snps):
        for j in range(i+1,num_snps):
            efi = effects[i]
            efj = effects[j]
            freqi = freqs[i]
            freqj = freqs[j]

            margxy = np.empty((3,3))
            for x in range(3):
                for y in range(3):
                    margxy[x,y] = probit_corr_xy(x,y, efi, efj, freqi, freqj, thresh) * freqi**x * (1-freqi)**(1-x) * freqj**y * (1-freqj)**(1-y)
                    if x==1:
                        margxy[x,y] *= 2
                    if y==1:
                        margxy[x,y] *= 2
            margxy /= np.sum(margxy)
            exy[i,j] = (margxy[1,1] + 2*margxy[1,2] + 2*margxy[2,1] + 4*margxy[2,2])


    # calculate expected correlation
    rho = np.empty((num_snps,num_snps))
    for i in range(num_snps):
        rho[i,i] = 1.0
        for j in range(i+1,num_snps):
            rho[i,j] = (exy[i,j] - ex[i] * ex[j]) / np.sqrt(ex2[i] - ex[i]**2) / np.sqrt(ex2[j] - ex[j]**2)
            rho[j,i] = rho[i,j]
    return rho, ex











def heterogeneity(cases, conts):
    num_snps = cases.shape[1]
    N = cases.shape[0]
    Np = conts.shape[0]
    z = np.zeros(num_snps)
    R = np.corrcoef(cases.T)
    Rp = np.corrcoef(conts.T)
    Y = np.sqrt(N*Np/(N+Np)) * (R-Rp)
    for i in range(num_snps):
        p_case = np.sum(cases[:,i]) / (2*cases.shape[0])
        p_control = np.sum(conts[:,i]) / (2*conts.shape[0])
        gamma = p_case/(1-p_case) / (p_control/(1-p_control))
        z[i] = np.sqrt(p_control*(1-p_control)) * (gamma-1) / (p_control*(gamma-1) + 1)


    numer = 0.0
    denom = 0.0
    for i in range(num_snps):
        for j in range(i+1,num_snps):
            wij = z[i] * z[j]
            yij = Y[i,j]
            numer += wij * yij
            denom += wij * wij
    score = numer / np.sqrt(denom)
    return score

def heterogeneity_expected_corr(ncases, nconts, effects, thresh, freqs, heritability, verbose=False):
    num_snps = len(effects)
    N = ncases
    Np = nconts

    R, ex = expected_corr_unnorm(effects, thresh, freqs, heritability, verbose=False)
    Rp = np.eye(num_snps)
    Y = np.sqrt(N*Np/(N+Np)) * (R-Rp)

    z = np.zeros(num_snps)
    for i in range(num_snps):
        p_case = ex[i]/2
        p_control = freqs[i]
        gamma = p_case/(1-p_case) / (p_control/(1-p_control))
        z[i] = np.sqrt(p_control*(1-p_control)) * (gamma-1) / (p_control*(gamma-1) + 1)


    numer = 0.0
    denom = 0.0
    for i in range(num_snps):
        for j in range(i+1,num_snps):
            wij = z[i] * z[j]
            yij = Y[i,j]
            numer += wij * yij
            denom += wij * wij
    score = numer / np.sqrt(denom)
    return score


def generate_snp_props(num_snps, ps, h_sq):
    beta_val = np.sqrt(h_sq / np.sum(2 * np.multiply(ps,1-ps)))
    betas = np.array([beta_val]*num_snps)
    return ps, betas

def generate_homhet_cohort(num_cases, num_conts, num_snps, ps, h_sq, het=False):
    #num_cases = 5000
    #num_conts = 5000
    #num_snps = 10
    prev = 0.01
    thresh = norm.ppf(1-prev, loc=0, scale=1)
    #h_sq = 0.1

    ps, betas = generate_snp_props(num_snps, ps, h_sq)
    cases, conts = generate_cohort(num_cases=num_cases,
                                   num_conts=num_conts,
                                   freqs=ps,
                                   betas=betas,
                                   h_sq=h_sq,
                                   thresh=thresh)
    if het:
        pi = 0.5
        conts_temp = np.random.binomial(n=2, p=ps, size=(num_conts,num_snps))
        cases = np.concatenate((cases[:int(num_cases * pi)], conts_temp[:int(num_cases * (1-pi))]),axis=0)
    return cases,conts

def generate_controls(num_cases, num_conts, num_snps, ps, h_sq):
    cases = np.random.binomial(n=2, p=ps, size=(num_cases,num_snps))
    conts = np.random.binomial(n=2, p=ps, size=(num_conts,num_snps))
    return cases,conts

