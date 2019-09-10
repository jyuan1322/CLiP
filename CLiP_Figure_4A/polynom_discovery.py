import numpy as np
import sys, argparse
import matplotlib.pyplot as plt
from pprint import pprint
sys.path.append('../')
from CLiPY import generate_population, generate_pss_model_simple, run_cont_heterogeneity_on_pop

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

def run(num_inds=50000, num_snps=10, h=0.1, deg=8, numtrain=5, symmetric=False, verbose=False):

    # Sample data
    training_data = []
    for i in range(numtrain):
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
        training_data.append({"independent_snps":independent_snps,
                              "independent_pop":independent_pop,
                              "hetero_snps":hetero_snps,
                              "hetero_pop":hetero_pop})



    plt.ion()
    count = 0
    plot_count = 1
    plot_cnt_tot = 10

    ## when learn_coef=True, coefficients are learned directly rather than roots
    ## -- This allows coefficients for odd degrees to be set to 0
    ## -- Not all polynomials have maximum number of roots

    # initialization of weight distribution here
    coef_wts = np.random.normal(loc=0,scale=500,size=deg+1)
    if symmetric:
        coef_wts[1::2] = 0

    while not valid_poly(coef_wts, symmetric=symmetric,
                         minval=-0.5, maxval=0.5):
        coef_wts = np.random.normal(loc=0,scale=500,size=deg+1)
        if symmetric:
            coef_wts[1::2] = 0
    scrhoms = []
    scrhets = []
    for i in range(numtrain):
        independent_pop = training_data[i]["independent_pop"]
        independent_snps = training_data[i]["independent_snps"]
        hetero_pop = training_data[i]["hetero_pop"]
        hetero_snps = training_data[i]["hetero_snps"]
        HetScorehom = run_cont_heterogeneity_on_pop(independent_pop,
                                          independent_snps,
                                          weight_func=None,
                                          coef_wts=coef_wts,
                                          symmetric=symmetric)
        HetScorehet = run_cont_heterogeneity_on_pop(hetero_pop,
                                          hetero_snps,
                                          weight_func=None,
                                          coef_wts=coef_wts,
                                          symmetric=symmetric)
        scrhoms.append(HetScorehom)
        scrhets.append(HetScorehet)
    HetScorehom = np.mean(scrhoms)
    HetScorehet = np.mean(scrhets)

    # HetScorediff = HetScorehom
    HetScorediff = HetScorehet - HetScorehom

    if symmetric:
        pc_range = np.linspace(-0.5,0.5,500,endpoint=False)
    else:
        pc_range = np.linspace(0,1,500,endpoint=False)

    polynom = np.poly1d(coef_wts,r=True)
    if verbose:
        plt.scatter(pc_range,
                   [polynom(x) for x in pc_range],
                   color=str(1.0-plot_count/plot_cnt_tot))

    if verbose:
        plt.show(block=False)
        plt.pause(0.05)
    while count < 50:
        count += 1

        inc = np.random.randint(0,deg+1)
        if symmetric and inc % 2 ==1:
            while inc % 2 == 1:
                inc = np.random.randint(0,deg+1)
        coef_wts_cand = np.copy(coef_wts)
        coef_wts_cand[inc] += np.random.normal(loc=0, scale=500)
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
        HetScorehom_cand = run_cont_heterogeneity_on_pop(independent_pop,
                                               independent_snps,
                                               weight_func=None,
                                               coef_wts=coef_wts_cand,
                                               symmetric=symmetric)
        HetScorehet_cand = run_cont_heterogeneity_on_pop(hetero_pop,
                                               hetero_snps,
                                               weight_func=None,
                                               coef_wts=coef_wts_cand,
                                               symmetric=symmetric)
        scrhoms = []
        scrhets = []
        for i in range(numtrain):
            independent_pop = training_data[i]["independent_pop"]
            independent_snps = training_data[i]["independent_snps"]
            hetero_pop = training_data[i]["hetero_pop"]
            hetero_snps = training_data[i]["hetero_snps"]
            HetScorehom_cand = run_cont_heterogeneity_on_pop(independent_pop,
                                              independent_snps,
                                              weight_func=None,
                                              coef_wts=coef_wts_cand,
                                              symmetric=symmetric)
            HetScorehet_cand = run_cont_heterogeneity_on_pop(hetero_pop,
                                              hetero_snps,
                                              weight_func=None,
                                              coef_wts=coef_wts_cand,
                                              symmetric=symmetric)
            scrhoms.append(HetScorehom_cand)
            scrhets.append(HetScorehet_cand)
        HetScorehom_cand = np.mean(scrhoms)
        HetScorehet_cand = np.mean(scrhets)

        # HetScorediff_cand = HetScorehom_cand
        HetScorediff_cand = HetScorehet_cand - HetScorehom_cand

        print("count: ", count)
        print("HetScorediff:", HetScorediff)
        print("HetScorediff_cand: ", HetScorediff_cand)
        print("coef_wts:", coef_wts)
        print("-"*20)
        if HetScorediff_cand > HetScorediff:
            HetScorediff = HetScorediff_cand
            coef_wts = coef_wts_cand
            # plot updated value here
            polynom = np.poly1d(coef_wts, r=False)
            if verbose:
                plt.figure()
                plt.scatter(pc_range,
                           [polynom(x) for x in pc_range],
                           color=str(1.0-plot_count/plot_cnt_tot))
                plt.xlabel("PRS percentile")
                plt.ylabel("individual weight phi")
                plt.show(block=False)
                plt.pause(0.05)
            plot_count = min(plot_count+1, plot_cnt_tot)

    return HetScorediff, coef_wts


def test_funcs(filename, numtrain=20, num_snps=10, h=0.1, num_inds=100000, symmetric=False, verbose=True):
    training_data = []
    for i in range(numtrain):
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
        training_data.append({"independent_snps":independent_snps,
                              "independent_pop":independent_pop,
                              "hetero_snps":hetero_snps,
                              "hetero_pop":hetero_pop})

    # test all polynoms in file
    coef_cands = []
    HetScorecands = []
    HetScorecand_stds = []
    count = 1
    with open(filename) as f:
        for line in f:
            line = line.split(":")[1]
            coef_wts = [float(x) for x in line.split(",")]

            scr_diffs = []
            for i in range(numtrain):
                independent_pop = training_data[i]["independent_pop"]
                independent_snps = training_data[i]["independent_snps"]
                hetero_pop = training_data[i]["hetero_pop"]
                hetero_snps = training_data[i]["hetero_snps"]
                HetScorehom = run_cont_heterogeneity_on_pop(independent_pop,
                                                  independent_snps,
                                                  weight_func=None,
                                                  coef_wts=coef_wts,
                                                  symmetric=symmetric)
                HetScorehet = run_cont_heterogeneity_on_pop(hetero_pop,
                                                  hetero_snps,
                                                  weight_func=None,
                                                  coef_wts=coef_wts,
                                                  symmetric=symmetric)
                HetScorediff = HetScorehet - HetScorehom
                scr_diffs.append(HetScorediff)
            coef_cands.append(coef_wts)
            HetScorecands.append(np.mean(scr_diffs))
            HetScorecand_stds.append(np.std(scr_diffs))
            print(count, HetScorecands[-1], HetScorecand_stds[-1])
            with open("evaluated_candidates.txt", "a") as f:
                f.write("%s|%s|%s|%s\n" % (count, HetScorecands[-1], HetScorecand_stds[-1], filename))
            count += 1

    if verbose:
        plt.errorbar(range(len(coef_cands)), HetScorecands, yerr=HetScorecand_stds)
        plt.show()


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Discover optimal quantitative PRS weighting function to distinguish homogeneous and heterogeneous cohorts')
    parser.add_argument('--test', '-t', dest='test', action='store_true')
    args = parser.parse_args()
    
    if args.test:
         test_funcs("discovered_polynomials.txt", numtrain=20, verbose=False)
    else:
        count = 0
        while count < 50:
            HetScorediff, coef_wts = run(num_inds=5000, num_snps=10, h=0.1, deg=2, numtrain=5, symmetric=False, verbose=False)
            print(HetScorediff, coef_wts)
            if HetScorediff > 4:
                count += 1
                with open("discovered_polynomials.txt", "a") as myfile:
                    myfile.write(str(HetScorediff) + ": " + ",".join([str(x) for x in coef_wts]) + "\n")
