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
sys.path.append('../')
from CLiPY_utils import generate_pss_model_simple, \
                        calc_var_from_geno, \
                        generate_population
from CLiPY import run_cont_heterogeneity_on_pop

def valid_poly(coefs, **kwargs):
    # check that this polynomial is positive over the interval:
    # has noroots in the span of 0, 1, and f(0.5) > 0
    symmetric = False
    if "symmetric" in kwargs:
        symmetric = kwargs["symmetric"]
        minval = kwargs["minval"]
        maxval = kwargs["maxval"]

    polynom = np.poly1d(coefs,r=False)

    if symmetric:
        for c in polynom.r:
            if c > minval and c < maxval:
                return False
        if polynom(0.0) <= 0:
            return False
        else:
            return True
    else:
        for c in polynom.r:
            if c > 0 and c < 1:
                return False
        if polynom(0.5) <= 0:
            return False
        else:
            return True


def learn_weight_func(num_snps=10, h=0.1, num_inds=5000):
    """
    Evaluate quantitative phenotype score for multiple values of case sample size
    """
    independent_snps = generate_pss_model_simple(num_snps, h)
    hetero_snps = generate_pss_model_simple(num_snps, h)

    independent_pop = generate_population(independent_snps, num_inds=num_inds, h=h)
    hetero_pop = generate_population(hetero_snps, num_inds=num_inds, h=h)

    # mean-center phenos
    independent_pop = (independent_pop[0], np.array(independent_pop[1]) - np.mean(independent_pop[1]))
    hetero_pop = (hetero_pop[0], np.array(hetero_pop[1]) - np.mean(hetero_pop[1]))

    het_mean = np.mean(hetero_pop[1])
    het_std = np.std(hetero_pop[1])
    hetero_pop[1][int(num_inds/2):] = np.random.normal(loc=het_mean,
                                                       scale=het_std,
                                                       size=int(num_inds/2))



    ###########################
    ### Test to find best w ###
    ###########################
    learn_coefs = False
    symmetric = False
    if not learn_coefs: # don't apply symmetry to the bin weights method
        symmetric = False

    plt.ion()
    count = 0
    plot_count = 1
    plot_cnt_tot = 50 if not learn_coefs else 10
    deg = 4

    ## when learn_coef=True, coefficients are learned directly rather than roots
    ## -- This allows coefficients for odd degrees to be set to 0
    ## -- Not all polynomials have maximum number of roots

    # initialization of weight distribution here
    if learn_coefs:
        coef_wts = np.random.normal(loc=0,scale=10,size=deg+1)
        if symmetric:
            coef_wts[1::2] = 0

        while not valid_poly(coef_wts, symmetric=symmetric,
                             minval=-0.5, maxval=0.5):
            coef_wts = np.random.normal(loc=0,scale=1,size=deg+1)
            if symmetric:
                coef_wts[1::2] = 0
        HSC_hom = run_cont_heterogeneity_on_pop(independent_pop,
                                          independent_snps,
                                          weight_func=None,
                                          coef_wts=coef_wts,
                                          symmetric=symmetric)
        HSC_het = run_cont_heterogeneity_on_pop(hetero_pop,
                                          hetero_snps,
                                          weight_func=None,
                                          coef_wts=coef_wts,
                                          symmetric=symmetric)
    else: # block coef
        numbins = 40
        block_wts = np.array([0.0]*int(numbins/2) + [1.0]*int(numbins/2))
        # block_wts = np.random.normal(loc=1.0, scale=0.01, size=numbins)

        HSC_hom = run_cont_heterogeneity_on_pop(independent_pop,
                                          independent_snps,
                                          weight_func=None,
                                          block_wts=block_wts,
                                          numbins=numbins)
        HSC_het = run_cont_heterogeneity_on_pop(hetero_pop,
                                          hetero_snps,
                                          weight_func=None,
                                          block_wts=block_wts,
                                          numbins=numbins)
    # HSC_diff = HSC_hom
    HSC_diff = HSC_het - HSC_hom

    if symmetric:
        # pc_range = np.linspace(np.amin(independent_pop[1]), np.amax(independent_pop[1]), 500)
        pc_range = np.linspace(-0.5,0.5,500,endpoint=False)
    else:
        pc_range = np.linspace(0,1,500,endpoint=False)

    if learn_coefs:
        polynom = np.poly1d(coef_wts,r=True)
        plt.scatter(pc_range,
                   [polynom(x) for x in pc_range],
                   color=str(1.0-plot_count/plot_cnt_tot))
    else:
        plt.scatter(pc_range,
                    block_wts[[int(x) for x in np.floor((pc_range) * numbins)]],
                    color=str(1.0-plot_count/plot_cnt_tot))
    plt.show(block=False)
    plt.pause(0.05)
    if symmetric:
        plt.figure()
    while True:
        count += 1

        if learn_coefs:
            inc = np.random.randint(0,deg+1)
            if symmetric and inc % 2 ==1:
                while inc % 2 == 1:
                    inc = np.random.randint(0,deg+1)
            coef_wts_cand = np.copy(coef_wts)
            coef_wts_cand[inc] += np.random.normal(loc=0, scale=10)
            # check that this is valid
            while not valid_poly(coef_wts_cand, symmetric=symmetric,
                                 minval=-0.5,
                                 maxval=0.5):
                print(".",)
                inc = np.random.randint(0,deg+1)
                if symmetric and inc % 2 ==1:
                    while inc % 2 == 1:
                        inc = np.random.randint(0,deg+1)
                coef_wts_cand = np.copy(coef_wts)
                coef_wts_cand[inc] += np.random.normal(loc=0, scale=10)
            HSC_hom_cand = run_cont_heterogeneity_on_pop(independent_pop,
                                                   independent_snps,
                                                   weight_func=None,
                                                   coef_wts=coef_wts_cand,
                                                   symmetric=symmetric)
            HSC_het_cand = run_cont_heterogeneity_on_pop(hetero_pop,
                                                   hetero_snps,
                                                   weight_func=None,
                                                   coef_wts=coef_wts_cand,
                                                   symmetric=symmetric)
        else:
            inc = np.random.randint(0, numbins)
            block_wts_cand = np.copy(block_wts)

            candidate = block_wts_cand[inc] + np.random.normal(loc=0, scale=0.01)
            if candidate <0:
                continue
            block_wts_cand[inc] = candidate

            HSC_hom_cand = run_cont_heterogeneity_on_pop(independent_pop,
                                                   independent_snps,
                                                   weight_func=None,
                                                   block_wts=block_wts_cand,
                                                   numbins=numbins)
            HSC_het_cand = run_cont_heterogeneity_on_pop(hetero_pop,
                                                   hetero_snps,
                                                   weight_func=None,
                                                   block_wts=block_wts_cand,
                                                   numbins=numbins)
        # HSC_diff_cand = HSC_hom_cand
        HSC_diff_cand = HSC_het_cand - HSC_hom_cand
        if learn_coefs:
            print("count: ", count)
            print("HSC_diff:", HSC_diff)
            print("HSC_diff_cand: ", HSC_diff_cand)
            print("coef_wts:", coef_wts)
            print("-"*20)
        if HSC_diff_cand > HSC_diff:
            HSC_diff = HSC_diff_cand
            if learn_coefs:
                coef_wts = coef_wts_cand
                # plot updated value here
                polynom = np.poly1d(coef_wts, r=False)
                plt.scatter(pc_range,
                           [polynom(x) for x in pc_range],
                           color=str(1.0-plot_count/plot_cnt_tot))
                plt.xlabel("PRS percentile")
                plt.ylabel("individual weight phi")
                plt.show(block=False)
                plt.pause(0.05)
                plot_count = min(plot_count+1, plot_cnt_tot)
            else:
                block_wts = block_wts_cand

        if not learn_coefs and count > 500:
            count = 0
            if plot_count < plot_cnt_tot:
                plot_count += 1
            print("-"*20)
            print("HSC diff:", HSC_diff)

            coefs = np.polyfit(np.linspace(0,1,numbins),
                               block_wts,
                               deg=deg)
            print("poly coefs:", coefs)

            # score from the poly coefs directly
            try:
                HSC_hom_cand = run_cont_heterogeneity_on_pop(independent_pop,
                                                       independent_snps,
                                                       weight_func=None,
                                                       coef_wts=coefs)
                HSC_het_cand = run_cont_heterogeneity_on_pop(hetero_pop,
                                                       hetero_snps,
                                                       weight_func=None,
                                                       coef_wts=coefs)
                HSC_diff_cand = HSC_het_cand - HSC_hom_cand
                print("poly coef HSC:", HSC_diff_cand)
            except:
                print("poly coef HSC: invalid")
            plt.scatter(pc_range, block_wts[[int(x) for x in np.floor((pc_range) * numbins)]],
                        color=str(1.0-plot_count/plot_cnt_tot))
            polynom = np.poly1d(coefs)
            plt.plot(np.linspace(0,1,numbins), polynom(np.linspace(0,1,numbins)),
                                 color=str(1.0-plot_count/plot_cnt_tot))
            plt.xlabel("PRS percentile")
            plt.ylabel("individual weight phi")
            plt.show(block=False)
            plt.pause(0.05)



if __name__=="__main__":
    learn_weight_func()
