import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle, sys, argparse
from scipy.stats import norm
sys.path.append('../')
from CLiP import generate_cohort, \
                 generate_cohort_splits, \
                 generate_snps_splits, \
                 heterogeneity

def test_n_splits(num_sub_phenos, frac_shared_effects, num_snps=100, num_cases=5000, num_conts=5000):
    num_snps_shared = num_snps * frac_shared_effects
    # ensure equal number of non-shared SNPs across sub-phenotypes
    assert (num_snps - num_snps_shared) % num_sub_phenos == 0
    num_snps_exclsv = (num_snps - num_snps_shared)/num_sub_phenos
    h_sq = 0.034
    ps = np.array([0.2]*num_snps)

    cases = np.zeros((0,num_snps))
    conts = np.zeros((0,num_snps))
    for i in range(num_sub_phenos):
        num_sub_cases = int(num_cases / num_sub_phenos)
        num_sub_conts = num_sub_cases
        sub_betas = np.array([ int(j < num_snps_shared or (
                                   j>= num_snps_shared + i*(num_snps_exclsv) and \
                                   j< num_snps_shared + (i+1)*(num_snps_exclsv))
                               ) for j in range(num_snps)])
        print(num_snps_shared)
        print(num_snps_exclsv)
        print(sub_betas)

        # set variance explained over the subset
        beta_val = np.sqrt(h_sq / np.sum(np.multiply(2 * np.multiply(ps,1-ps),
                                              sub_betas)))
        sub_betas = beta_val * sub_betas
        print(i, num_sub_cases)
        print(sub_betas)
        print("subset variance explained:", np.dot(np.square(sub_betas), 2*np.multiply(ps,1-ps)))

        # generate sub-cohort
        prev = 0.01
        thresh = norm.ppf(1-prev, loc=0, scale=1)
        sub_cases, sub_conts = generate_cohort(num_cases=num_sub_cases,
                                               num_conts=num_sub_conts,
                                               freqs=ps,
                                               betas=sub_betas,
                                               h_sq=h_sq,
                                               thresh=thresh)
        cases = np.concatenate((cases, sub_cases), axis=0)
        conts = np.concatenate((conts, sub_conts), axis=0)
    score = heterogeneity(cases, conts)
    return score

def plot_grid(results):
    # num_subpheno_list = [1,2,3,4,5,6,7,8]
    num_subpheno_list = [1,2,4,8]
    frac_shared_effects_list = [0, .25, .5, .75, 1]

    colors = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(
                                        vmin=min(num_subpheno_list),
                                        vmax=max(num_subpheno_list)),
                                        cmap='cool')
    colors.set_array(num_subpheno_list)

    fig,ax = plt.subplots(1,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i,nsubph in enumerate(num_subpheno_list):
        means = []
        stds = []
        for j,fsheff in enumerate(frac_shared_effects_list):
            mean = np.mean(results[nsubph][fsheff])
            std = np.std(results[nsubph][fsheff])
            means.append(mean)
            stds.append(std)
        plt.errorbar(1 - np.array(frac_shared_effects_list), means, 
                                  yerr=stds, c=colors.to_rgba(nsubph), label=nsubph, capsize=5)
    # plt.legend(frameon=False)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.xlabel("Fraction of SNP effects unique to sub-phenotype")
    plt.ylabel("Heterogeneity Score")
    plt.savefig("table1_disjoint_subphenos.png", format="png", dpi=1000)
    plt.show()

if __name__=="__main__":
    if os.path.exists("split_results.p"):
        results = pickle.load(open("split_results.p", "rb"))
        num_subpheno_list = [1,2,3,4,5,6,7,8]
        frac_shared_effects_list = [0, .25, .5, .75, 1]
        # num_subpheno_list = [2,3]
        # frac_shared_effects_list = [0, .5, 1]
        print("|     ", end="")
        print("     |     ".join([str(x) for x in frac_shared_effects_list]))
        for i,nsubph in enumerate(num_subpheno_list):
            print("|", end="")
            for j,fsheff in enumerate(frac_shared_effects_list):
                mean = np.mean(results[nsubph][fsheff])
                std = np.std(results[nsubph][fsheff])
                print("%0.2f, %0.2f|" % (mean, std), end="")
            print("\n")

        print("\n"*20)
        for i,nsubph in enumerate(num_subpheno_list):
            print("%s" % nsubph, end="")
            for j,fsheff in enumerate(frac_shared_effects_list):
                mean = np.mean(results[nsubph][fsheff])
                std = np.std(results[nsubph][fsheff])
                print(" & $%0.2f \\pm %0.2f$" % (mean, std), end="")
            print("\\\\")
            print("\\hline")
        plot_grid(results)
    else:
        num_cases = 50000
        num_snps = 100
        ps = np.array([0.2]*num_snps)
        num_trials = 20
        # num_subpheno_list = [1,2,3,4,5,6,7,8]
        num_subpheno_list = [2,3]
        # frac_shared_effects_list = [0, .25, .5, .75, 1]
        frac_shared_effects_list = [0, .5, 1]

        split_results = {}
        for i in num_subpheno_list:
            split_results[i] = {}
            for j in frac_shared_effects_list:
                split_results[i][j] = []
        for nsubph in num_subpheno_list:
            for fsheff in frac_shared_effects_list:
                for nt in range(num_trials):
                    print(nsubph, fsheff, nt)
                    h_sq = 0.05
                    sub_betas_list, ps = generate_snps_splits(num_sub_phenos=nsubph,
                                                              frac_shared_effects=fsheff,
                                                              num_snps=num_snps,
                                                              ps=ps,
                                                              h_sq=h_sq)

                    cases, conts = generate_cohort_splits(num_sub_phenos=nsubph,
                                                          sub_betas_list=sub_betas_list,
                                                          ps=ps,
                                                          num_cases=num_cases,
                                                          num_conts=num_cases,
                                                          h_sq=h_sq)
                    score = heterogeneity(cases,conts)
                    split_results[nsubph][fsheff].append(score)
        pickle.dump(split_results, open("split_results.p", "wb"))

