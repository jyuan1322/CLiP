import numpy as np
import sys, os
from scipy import integrate
from scipy.stats import norm, rankdata
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
sys.path.append('../')
from CLiPY_utils import generate_pss_model_simple, generate_population
from CLiPY import corr

def generate_heatmap(file_out,numbins=20, subset=None):
    num_snps = 10
    N = 50000
    h = 0.5
    num_trials = 5
    if subset is None:
        Tps = np.linspace(0,1,numbins)
        Tns = np.linspace(0,1,numbins)
    else:
        Tps = np.linspace(0, 1-subset, numbins)
        Tns = np.linspace(subset, 1, numbins)
    a = np.empty((numbins, numbins))
    for tn,Tneg in enumerate(Tns):
        for tp,Tpos in enumerate(Tps):
            scores = []
            for nt in range(num_trials):
                independent_snps = generate_pss_model_simple(num_snps, h)
                independent_pop = generate_population(independent_snps, num_inds=N, h=h)

                # mean-center phenos
                independent_pop = (independent_pop[0], np.array(independent_pop[1]) - np.mean(independent_pop[1]))
                genos, phenos = independent_pop
                phenos = norm.cdf(phenos, loc=np.mean(phenos), scale=np.std(phenos))
                weights = np.zeros(N)
                for i,ph in enumerate(phenos):
                    if ph < Tneg:
                        weights[i] += 1
                    if ph > Tpos:
                        weights[i] += 1
                weights /= np.sum(weights)
                Rnull = np.corrcoef(genos.T)
                R = corr(genos.T, weights)

                iu = np.triu_indices(genos.shape[1], 1)
                Rave = np.mean(R[iu])
                Rnullave = np.mean(Rnull[iu])

                w2 = np.sum(weights**2)
                w2 = 1/np.sqrt(w2 - 1/N)

                score = w2*(Rave-Rnullave)
                scores.append(score)
                print(tn, tp, score)
            a[tn,tp] = np.mean(scores)

    with open(file_out, "wb") as f:
        pickle.dump(a, f)

def generate_plot(numbins, subset, fname):
    if not os.path.exists(fname):
        generate_heatmap(fname, numbins=numbins+1, subset=subset)

    with open(fname, "rb") as f:
        a = pickle.load(f)
    ax = sns.heatmap(a, linewidth=0.0, center=0, cmap="RdBu")
    ax.invert_yaxis()
    ax.set_xticks(range(1, numbins+2))
    ax.set_yticks(range(1, numbins+2))

    if subset is None:
        intervals = [str(x) for x in np.around(np.linspace(0,1,numbins+1),decimals=2)]
        print(intervals)
        for i,itv in enumerate(intervals):
            if float(itv) * 100 % 10 != 0 and float(itv) != 0 and float(itv) != 1:
                intervals[i] = ""
            else:
                print(itv)
        ax.set_xticklabels(intervals)
        ax.set_yticklabels(intervals)
    else:
        intervalsx = [str(x) for x in np.around(np.linspace(0,1-subset,numbins+1),decimals=3)]
        intervalsy = [str(x) for x in np.around(np.linspace(subset,1,numbins+1),decimals=2)]
        print(intervalsx)
        for i,itv in enumerate(intervalsx):
            if float(itv) * 1000 % 10 != 0 and float(itv) != 0 and float(itv) != 1:
                intervalsx[i] = ""
                intervalsy[i] = ""
            else:
                print(itv)
        ax.set_xticklabels(intervalsx)
        ax.set_yticklabels(intervalsy)

    ax.set_xlabel("Increasing PRS Indicator Function", fontsize=15)
    ax.set_ylabel("Decreasing PRS Indicator Function", fontsize=15)
    
    if subset is None:
        plt.savefig("step_heatmap.eps", format="eps", dpi=300)
    else:
        plt.savefig("step_heatmap_subset.eps", format="eps", dpi=300)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    generate_plot(numbins=40, subset=None, fname="step_heatmap_full.pickle")
    generate_plot(numbins=40, subset=0.9, fname="step_heatmap_subset.pickle")
    
