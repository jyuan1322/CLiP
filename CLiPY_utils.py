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


def generate_pss_model_simple(num_snps, h):
    p = 0.5
    b_loc = np.sqrt(h/(num_snps * 2*p*(1-p)))
    b_scale = 0

    p = np.random.uniform(low=p,high=p, size=num_snps)
    beta = np.random.normal(loc=b_loc,scale=b_scale, size=num_snps)
    return p,beta

def calc_var_from_geno(snp_props):
    snp_ps, snp_betas = snp_props
    num_snps = snp_betas.shape
    return np.sum(snp_betas**2 * 2 * snp_ps * (1 - snp_ps), axis=0)

def generate_population(snps, num_inds=10000, h=1):
    """
    snps=[snp_ps, snp_betas]
    snp_ps: numpy length num_snps array with rafs
    snp_betas: numpy (num_snps, num phenos) matrix with betas
    """
    snp_ps, snp_betas = snps
    assert len(snp_ps) == len(snp_betas)
    num_snps = len(snp_ps)
    assert num_snps > 0
    num_phenos = 1

    # sample SNPs according to SNP props
    randoms = np.random.rand(num_inds, num_snps, 1)
    geno = np.random.binomial(n=2, p=snp_ps, size=(num_inds, num_snps))

    assert geno.shape == (num_inds, num_snps)
    pheno = np.dot(geno, snp_betas)
    genetic_var = calc_var_from_geno(snps)

    sigma = np.sqrt((1-h)/h * genetic_var)
    pheno = pheno + np.random.normal(loc=np.zeros(num_phenos), scale=sigma, size=num_inds)
    return geno, pheno

