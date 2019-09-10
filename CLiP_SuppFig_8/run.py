import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle, os, sys
from scipy.stats import norm
sys.path.append('../')
from CLiPX_utils import get_fixed_expression_params, generate_cohort


def artanh_correlation_sd(case_size_sets, heritability):
    num_snps = 100
    num_expr = 10
    g2wherit = 0.1

    freqs, gen2expr_wgtmat, alpha, g2wherit, heritability = get_fixed_expression_params(num_snps, num_expr, g2wherit, heritability)

    prev = 0.01
    thresh = norm.ppf(1-prev, loc=0, scale=1)

    expcorr, mu_case = expected_corr_unnorm(alpha,
                                            thresh, freqs, gen2expr_wgtmat,
                                            heritability=heritability)

    num_trials = 10
    std_means = []
    std_stds = []
    for cs in case_size_sets:
        stds = []
        for nt in range(num_trials):
            case_genos, cases = generate_cohort(num_snps, num_expr, gen2expr_wgtmat,
                                                        freqs, alpha, [1.0], [cs],
                                                        heritability, g2wherit,
                                                        thresh, use_logistic=False)
            corrs = np.corrcoef(cases[0].T)
            xvals = []
            yvals = []
            numvals = 0
            std = 0
            for a in range(num_expr):
                for b in range(a+1, num_expr):
                    xvals.append(expcorr[a,b])
                    yvals.append(corrs[a,b])
                    std += (expcorr[a,b] - corrs[a,b])**2
                    numvals += 1
            std = np.sqrt(std / (numvals-1))
            std = 0.5*np.log((1+std)/(1-std))
            stds.append(std)
        std_means.append(np.mean(stds))
        std_stds.append(np.std(stds))
    return std_means, std_stds



if __name__=="__main__":
    h_sqs = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2]
    case_size_sets = [100, 250, 500, 750, 1000, 2000]
    colors = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(
                                   vmin=min(h_sqs),
                                   vmax=max(h_sqs)),
                                   cmap='cool')
    colors.set_array(h_sqs)

    fig,ax = plt.subplots(1,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    cbar = fig.colorbar(colors,ticks=h_sqs)
    cbar.ax.set_yticklabels([str(x) for x in h_sqs])
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(10)
    if os.path.exists("artanh_corr_vals.p"):
        artanh_corr_vals = pickle.load(open("artanh_corr_vals.p", "rb"))
    else:
        artanh_corr_vals = {}

        for h_sq in h_sqs:
            std_means, std_stds = artanh_correlation_sd(case_size_sets, h_sq)
            artanh_corr_vals[h_sq] = (std_means,std_stds)
        pickle.dump(artanh_corr_vals, open("artanh_corr_vals.p", "wb"))

    for h_sq in artanh_corr_vals:
        std_means,std_stds = artanh_corr_vals[h_sq]
        plt.errorbar(case_size_sets, std_means, yerr=std_stds,
                     c=colors.to_rgba(h_sq), label=h_sqs, capsize=5)



    case_counts = np.array(range(50, 2200))
    plt.plot(case_counts, 1.0/np.sqrt(case_counts - 3), c='k', linestyle='--', linewidth=2, zorder=10)


    plt.xlabel("Cases sample size")
    plt.ylabel(r'$artanh(\sigma(r))$')
    plt.savefig("sample_corr_test_std_errs.eps", format="eps", dpi=1000)
    plt.show()

