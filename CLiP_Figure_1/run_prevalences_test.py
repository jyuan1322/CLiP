import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import norm
from pprint import pprint
sys.path.append('../')
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from run import convertORs

np.random.seed(0)

def test_prevalence():
    num_snps = 10
    freqs = np.array([0.5] * num_snps)
    prev = 0.01
    ORlist = [1.05, 1.10, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6]
    num_trials = 10
    subset_size = 100000

    cm = LinearSegmentedColormap.from_list("red_blue", ["red","blue"], N=100)
    colors_rg = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min(ORlist),
                                       vmax=max(ORlist)), cmap=cm)
    colors_rg.set_array(range(len(ORlist)))
    fig, ax = plt.subplots()

    for i in range(len(ORlist)):
        pc_case_logits = []
        pc_case_liabs = []

        ORs = np.array([ORlist[i]] * num_snps).T

        const = -np.log(1/prev - 1)
        beta_logits = np.log(ORs)

        thresh = norm.ppf(1-prev, loc=0, scale=1)
        betas = convertORs(ORs, prev)
        h_sq = np.dot(np.square(betas.flatten()),
                      2*np.multiply(freqs, 1-freqs))

        # sample logistic
        for nt in range(num_trials):
            # the same cohort used for both mdoels
            case_samples = np.random.binomial(n=2, p=freqs, size=(subset_size,num_snps))

            exp_score = 2*np.dot(beta_logits, freqs)
            scores = np.dot(case_samples, beta_logits) + \
                     const - \
                     exp_score # set mean score to 0
            scores = 1.0/(1.0+np.exp(-scores))
            case_labels = np.random.binomial(n=1, p=scores, size=subset_size)
            pc_case_logit = np.sum(case_labels) / len(case_labels)
            pc_case_logits.append(pc_case_logit)

            # sample liability

            # case_samples = np.random.binomial(n=2, p=freqs, size=(subset_size,num_snps))
            exp_score = 2*np.dot(betas, freqs)
            scores = np.dot(case_samples, betas) + \
                     np.random.normal(0,np.sqrt(1-h_sq),subset_size) - \
                     exp_score # set mean score to 0
            idxs = np.where(scores > thresh)[0]
            pc_case_liab = len(idxs) / len(scores)
            pc_case_liabs.append(pc_case_liab)

            print(pc_case_logit, pc_case_liab)

        plt.scatter(pc_case_logits, pc_case_liabs, c=np.array([colors_rg.to_rgba(ORlist[i])]))

    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
           ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel("Logistic prevalences")
    plt.xticks(rotation=45)
    plt.ylabel("Liability prevalences")
    # plt.ticklabel_format(style='sci', scilimits=(0,0))

    cbar = fig.colorbar(colors_rg,ticks=ORlist,cax=plt.axes([0.85, 0.11, 0.05, 0.77]))
    cbar.ax.set_yticklabels([str(x) for x in ORlist])
    plt.savefig("logistic_vs_liability_prevalences.eps", bbox_inches='tight', format="eps", dpi=500)
    plt.show()

if __name__=="__main__":
    test_prevalence()
