import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle, sys, argparse
from scipy.stats import norm
sys.path.append('../')
from CLiP import heterogeneity, generate_homhet_cohort


def plot_results(resultsall):
    ncases_list = [5000, 10000, 25000, 50000, 100000, 250000, 500000]
    split_fracs = [0, 0.25, 0.5, 0.75, 1]
    h_sq = 0.05
    results = resultsall[h_sq]

    ncases_list_log = np.log(np.array(ncases_list))
    colors = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(
                                        vmin=min(ncases_list_log),
                                        vmax=max(ncases_list_log)),
                                        cmap='summer_r')
    colors.set_array(ncases_list_log)
    fig,ax = plt.subplots(1,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    cbar = fig.colorbar(colors,ticks=ncases_list_log)
    cbar.ax.set_yticklabels([str(x) for x in ncases_list])
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(10)
    for ncases in ncases_list:
        means = []
        stds = []
        for split_frac in split_fracs:
            mean = np.mean(results[ncases][split_frac])
            std = np.std(results[ncases][split_frac])
            means.append(mean)
            stds.append(std)
        plt.errorbar(split_fracs, means, yerr=stds,
                     label=ncases, capsize=5,
                     c=colors.to_rgba(np.log(ncases)))
    # plt.legend(frameon=False)
    plt.xlabel("Fraction of true cases")
    plt.ylabel("Heterogeneity Score")
    plt.savefig("table2_casecontrol_het.png", format="png", dpi=1000)
    plt.show()

if __name__=="__main__":
    h_sq_list_all = [.01, .025, .05, .075, 0.1]
    h_sq_list = [.01]
    ncases_list = [5000, 10000, 25000, 50000, 100000, 250000, 500000]
    split_fracs = [0, 0.25, 0.5, 0.75, 1]

    num_snps = 100
    fixed_ps = np.array([0.2]*num_snps)
    num_trials = 20

    if os.path.exists("splitfrac_results_0.05.p"):
        # for picklefile in ["splitfrac_results_0.01.p"]:
        resultsall = {}
        for h_sq in h_sq_list_all:
            picklefile = "splitfrac_results_%s.p" % (h_sq)
            results = pickle.load(open(picklefile, "rb"))
            resultsall.update(results)
            for ncases in ncases_list:
                print("h_sq: %s, ncases: %s" % (h_sq, ncases))
                for split_frac in split_fracs:
                    mean = np.mean(results[h_sq][ncases][split_frac])
                    std = np.std(results[h_sq][ncases][split_frac])
                    print("\t splitfrac: %s, mean: %s, std: %s" % (split_frac, mean, std))

        print("\n"*10)
        for ncases in ncases_list:
            print("\\begin{tabular}{c|r}")
            print("\\multirow{5}{*}{%s} & 0 \\\\" % (ncases))
            print("& 0.25 \\\\")
            print("& 0.5 \\\\")
            print("& 0.75 \\\\")
            print("& 1.0 \\\\")
            print("\\end{tabular}")
            for h_sq in h_sq_list_all:
                print("& \\begin{tabular}{r}")
                for split_frac in split_fracs:
                    mean = np.mean(resultsall[h_sq][ncases][split_frac])
                    std = np.std(resultsall[h_sq][ncases][split_frac])
                    print("$%0.2f \pm %0.2f$\\\\" % (mean, std))
                print("\\end{tabular}")
            print("\\\\ \\hline")

        plot_results(resultsall)
    else:
        splitfrac_results = {}
        for i in h_sq_list:
            splitfrac_results[i] = {}
            for j in ncases_list:
                splitfrac_results[i][j] = {}
                for k in split_fracs:
                    splitfrac_results[i][j][k] = []
        for h_sq in h_sq_list:
            for ncases in ncases_list:
                for split_frac in split_fracs:
                    for nt in range(num_trials):
                        print("h_sq: %s, ncases: %s, split: %s, trial: %s" % (h_sq, ncases, split_frac, nt))
                        cases, conts = generate_homhet_cohort(num_cases=ncases,
                                                              num_conts=ncases,
                                                              num_snps=num_snps,
                                                              ps=fixed_ps,
                                                              h_sq=h_sq,
                                                              het=True,
                                                              pi=split_frac)
                        score = heterogeneity(cases,conts)
                        splitfrac_results[h_sq][ncases][split_frac].append(score)
        pickle.dump(splitfrac_results, open("splitfrac_results_%s.p" % h_sq_list[0], "wb"))
