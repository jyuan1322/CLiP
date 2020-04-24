import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle, sys, argparse
from scipy.stats import norm
# sys.path.append('../')
from CLiP import generate_snp_props, \
                 generate_cohort, \
                 heterogeneity, \
                 heterogeneity_expected_corr, \
                 generate_snps_splits
from logistic_vs_liability_test import generate_cohort_logistic, \
                                       heterogeneity_expected_corr_logit

def plot_results(h_sq_frac_range, results, hetsc_exp, outname):
    fig,ax = plt.subplots(1,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(h_sq_frac_range, [hetsc_exp]*len(h_sq_frac_range), label="expected homog",c='b')
    means = [np.mean(x) for x in results]
    stds = [np.std(x) for x in results]
    plt.errorbar(h_sq_frac_range, means, yerr=stds, label="heterog", capsize=5,c='g')

    # plt.legend()
    plt.xlabel(r'Fraction of subtype magnitude $\beta_2/\beta_1$')
    plt.ylabel("Heterogeneity Score")
    # plt.savefig(outname, format="png", dpi=500)
    plt.savefig(outname, format="eps", dpi=500)
    plt.show()

def run_liability():
    FILE_PATH = "subtype_frac_liab.p"
    if os.path.exists(FILE_PATH):
        h_sq_frac_range, results, hetsc_exp = pickle.load(open(FILE_PATH, "rb"))
        plot_results(h_sq_frac_range, results, hetsc_exp, outname="subtype_scale_liability.eps")
    else:
        num_cases = 30000
        num_conts = 30000
        num_snps = 100
        fixed_ps = np.array([0.2]*num_snps)
        # num_trials = 3
        num_trials = 20
        h_sq = 0.05
        h_sq_frac_range = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        sub_betas_list, _ = generate_snps_splits(num_sub_phenos=2,
                                                 frac_shared_effects=0,
                                                 num_snps=num_snps,
                                                 ps=fixed_ps,
                                                 h_sq=h_sq)

        # subtype 1 properties
        ps, betas = generate_snp_props(num_snps, fixed_ps, h_sq)
        prev = 0.01
        thresh = norm.ppf(1-prev, loc=0, scale=1)

        # expected score if all cases followed subtype 1
        hetsc_exp = heterogeneity_expected_corr(ncases = num_cases,
                                                nconts = num_cases,
                                                effects = betas,
                                                thresh = thresh,
                                                freqs = fixed_ps,
                                                heritability = h_sq, verbose=False)
        print(hetsc_exp)

        results = []
        for h_sq_frac in h_sq_frac_range:
            scores = []
            # generate subtype 2 properties
            # _, betas_sub2 = generate_snp_props(num_snps, fixed_ps, h_sq*h_sq_frac)
            sub_betas_list2, _ = generate_snps_splits(num_sub_phenos=2,
                                         frac_shared_effects=0,
                                         num_snps=num_snps,
                                         ps=fixed_ps,
                                         h_sq=h_sq*h_sq_frac)
            for nt in range(num_trials):
                print("h_sq_frac: %s, trial: %s" % (h_sq_frac, nt))
                cases_sub1, conts = generate_cohort(num_cases=int(num_cases/2),
                                                    num_conts=num_conts,
                                                    freqs=fixed_ps, # freqs=ps,
                                                    betas=sub_betas_list[0], # betas=betas,
                                                    h_sq=h_sq,
                                                    thresh=thresh)
                cases_sub2, _ = generate_cohort(num_cases=int(num_cases/2),
                                                num_conts=num_conts,
                                                freqs=fixed_ps, # freqs=ps,
                                                betas=sub_betas_list2[1], # betas=betas_sub2,
                                                h_sq=h_sq*h_sq_frac,
                                                thresh=thresh)
                cases = np.concatenate((cases_sub1, cases_sub2), axis=0)
                score = heterogeneity(cases,conts)
                scores.append(score)
            results.append(scores)

        pickle.dump((h_sq_frac_range, results, hetsc_exp), open(FILE_PATH, "wb"))



### Logistic Regression
def run_logistic():
    FILE_PATH = "subtype_frac_logit.p"
    if os.path.exists(FILE_PATH):
        h_sq_frac_range, results, hetsc_exp = pickle.load(open(FILE_PATH, "rb"))
        plot_results(h_sq_frac_range, results, hetsc_exp, outname="subtype_scale_logit.eps")
    else:
        num_cases = 30000
        num_conts = 30000
        num_snps = 100
        fixed_ps = np.array([0.2]*num_snps)
        # num_trials = 3
        num_trials = 20
        OR_val = 1.1
        OR_val_frac_range = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        prev = 0.01

        # subtype 1 properties
        # ORs = np.array([OR_val]*num_snps)
        ORs = np.array([OR_val]*int(num_snps/2) + [1.0]*int(num_snps/2))

        # expected score if all cases followed subtype 1
        hetsc_exp = heterogeneity_expected_corr_logit(ncases = num_cases,
                                                      nconts = num_cases,
                                                      ORs = ORs,
                                                      freqs = fixed_ps,
                                                      prev = prev, verbose=False)
        print(hetsc_exp)

        results = []
        for OR_val_frac in OR_val_frac_range:
            scores = []
            # generate subtype 2 properties
            ORs_frac_val = 1.0 + (OR_val - 1)*OR_val_frac
            # ORs_frac = np.array([ORs_frac_val]*num_snps)
            ORs2 = np.array([1.0]*int(num_snps/2) + [ORs_frac_val]*int(num_snps/2))
            for nt in range(num_trials):
                print("OR_val_frac: %s, trial: %s" % (OR_val_frac, nt))
                cases_sub1, conts = generate_cohort_logistic(num_cases = int(num_cases/2),
                                                             num_conts = num_conts,
                                                             freqs = fixed_ps,
                                                             ORs = ORs,
                                                             prev = prev)
                cases_sub2, _ = generate_cohort_logistic(num_cases = int(num_cases/2),
                                                         num_conts = num_conts,
                                                         freqs = fixed_ps,
                                                         ORs = ORs2,
                                                         prev = prev)
                cases = np.concatenate((cases_sub1, cases_sub2), axis=0)
                score = heterogeneity(cases,conts)
                scores.append(score)
            results.append(scores)

        pickle.dump((OR_val_frac_range, results, hetsc_exp), open(FILE_PATH, "wb"))



if __name__=="__main__":
    run_liability()
    run_logistic()
