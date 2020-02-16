import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle, sys, argparse
from scipy.stats import norm
sys.path.append('../')
from CLiP import generate_cohort, \
                 probit_corr, \
                 probit_corr_xy, \
                 expected_corr_unnorm, \
                 heterogeneity, \
                 heterogeneity_expected_corr, \
                 generate_snp_props, \
                 generate_homhet_cohort, \
                 generate_controls

def run_heritability():
    FILE_PATH = "simulate_results_varexp.p"
    fig, ax = plt.subplots(1,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if os.path.exists(FILE_PATH):
        pckl = pickle.load(open(FILE_PATH, "rb"))
        # hetsc_means_hom = pckl["hetsc_means_hom"]
        # hetsc_stds_hom = pckl["hetsc_stds_hom"]
        # hetsc_means_het = pckl["hetsc_means_het"]
        # hetsc_stds_het = pckl["hetsc_stds_het"]
        # hetsc_means_cont = pckl["hetsc_means_cont"]
        # hetsc_stds_cont = pckl["hetsc_stds_cont"]
        hetsc_means_hom = np.mean(pckl["hetscs_hom"])
        hetsc_stds_hom = np.std(pckl["hetscs_hom"])
        hetsc_means_het = np.mean(pckl["hetscs_het"])
        hetsc_stds_het = np.std(pckl["hetscs_het"])
        hetsc_means_cont = np.mean(pckl["hetscs_cont"])
        hetsc_stds_cont = np.std(pckl["hetscs_cont"])
        hetsc_exps = pckl["hetsc_exps"]
        h_sqs = pckl["h_sqs"]
        plt.errorbar(h_sqs, hetsc_means_hom, yerr=hetsc_stds_hom, color='red', capsize=5)
        plt.errorbar(h_sqs, hetsc_means_het, yerr=hetsc_stds_het, color='green', capsize=5)
        plt.errorbar(h_sqs, hetsc_means_cont, yerr=hetsc_stds_cont, color='black', capsize=5)
        plt.plot(h_sqs, hetsc_exps, color='blue')
        plt.xlabel("Variance explained by modeled SNPs")
        plt.ylabel("Heterogeneity Score")

        # label CLiP score
        xloc = 0.8 * (h_sqs[-1] - h_sqs[0])
        yloc1 = np.interp(xloc, h_sqs, hetsc_exps)
        yloc2 = np.interp(xloc, h_sqs, hetsc_means_het)
        ax.annotate(s='',xy=(xloc, yloc1), xycoords='data',
				    xytext=(xloc, yloc2),textcoords='data',
				    arrowprops=dict(arrowstyle="<->", color='gray'))
        ax.annotate(s='CLiP Score',xy=(xloc * 1.02, yloc1 + (yloc2-yloc1)*0.75), 
                    xycoords='data',fontsize=10.0,textcoords='data',
                    ha='left', color='gray')

        plt.savefig("simulate_basic_heritability.eps", format="eps", dpi=1000)
        plt.show()

    else:
        # hetsc_means_hom = []
        # hetsc_stds_hom = []
        # hetsc_means_het = []
        # hetsc_stds_het = []
        # hetsc_means_cont = []
        # hetsc_stds_cont = []
        hetscs_homs = []
        hetscs_hets = []
        hetscs_conts = []
        hetsc_exps = []
        h_sqs = [0.001, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
        num_trials = 20

        for h_sq in h_sqs:
            num_cases = 30000
            num_conts = 30000
            num_snps = 100
            fixed_ps = np.array([0.2]*num_snps)
            hetscs_hom = []
            hetscs_het = []
            hetscs_cont = []

            # expected score, homogeneous cases
            ps, betas = generate_snp_props(num_snps, fixed_ps, h_sq)
            prev = 0.01
            thresh = norm.ppf(1-prev, loc=0, scale=1)
            hetsc_exp = heterogeneity_expected_corr(ncases = num_cases,
                                        nconts = num_conts,
                                        effects = betas,
                                        thresh = thresh,
                                        freqs = ps,
                                        heritability = h_sq, verbose=False)
            hetsc_exps.append(hetsc_exp)

            for i in range(num_trials):
                print("h_sq: %s, trial: %s" % (h_sq, i))

                # homogeneous cases, controls
                cases,conts = generate_homhet_cohort(num_cases, num_conts, num_snps, ps, h_sq)
                score = heterogeneity(cases,conts)
                hetscs_hom.append(score)

                # heterogeneous cases, controls
                cases,conts = generate_homhet_cohort(num_cases, num_conts, num_snps, ps, h_sq, het=True)
                score = heterogeneity(cases,conts)
                hetscs_het.append(score)

                # controls, controls
                cases,conts = generate_controls(num_cases, num_conts, num_snps, ps, h_sq)
                score = heterogeneity(cases,conts)
                hetscs_cont.append(score)


            # hetsc_means_hom.append(np.mean(hetscs_hom))
            # hetsc_stds_hom.append(np.std(hetscs_hom))
            # hetsc_means_het.append(np.mean(hetscs_het))
            # hetsc_stds_het.append(np.std(hetscs_het))
            # hetsc_means_cont.append(np.mean(hetscs_cont))
            # hetsc_stds_cont.append(np.std(hetscs_cont))
            hetscs_homs.append(hetscs_hom)
            hetscs_hets.append(hetscs_het)
            hetscs_conts.append(hetscs_cont)
        pickle.dump({"hetscs_homs":hetscs_homs,
                     "hetscs_hets":hetscs_hets,
                     "hetscs_conts":hetscs_conts,
                     "hetsc_exps":hetsc_exps,
                     "h_sqs":h_sqs}, open(FILE_PATH, "wb"))
        """
        "hetsc_means_hom":hetsc_means_hom,
        "hetsc_stds_hom":hetsc_stds_hom,
        "hetsc_means_het":hetsc_means_het,
        "hetsc_stds_het":hetsc_stds_het,
        "hetsc_means_cont":hetsc_means_cont,
        "hetsc_stds_cont":hetsc_stds_cont,
        """
def run_sample_size():
    FILE_PATH = "simulate_results_samplesize.p"
    fig, ax = plt.subplots(1,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if os.path.exists(FILE_PATH):
        pckl = pickle.load(open(FILE_PATH, "rb"))
        # hetsc_means_hom = pckl["hetsc_means_hom"]
        # hetsc_stds_hom = pckl["hetsc_stds_hom"]
        # hetsc_means_het = pckl["hetsc_means_het"]
        # hetsc_stds_het = pckl["hetsc_stds_het"]
        # hetsc_means_cont = pckl["hetsc_means_cont"]
        # hetsc_stds_cont = pckl["hetsc_stds_cont"]
        # hetsc_exps = pckl["hetsc_exps"]
        hetsc_means_hom = np.mean(pckl["hetscs_hom"])
        hetsc_stds_hom = np.std(pckl["hetscs_hom"])
        hetsc_means_het = np.mean(pckl["hetscs_het"])
        hetsc_stds_het = np.std(pckl["hetscs_het"])
        hetsc_means_cont = np.mean(pckl["hetscs_cont"])
        hetsc_stds_cont = np.std(pckl["hetscs_cont"])
        sample_size = pckl["sample_size"]
        plt.errorbar(sample_size, hetsc_means_hom, yerr=hetsc_stds_hom, color='red', capsize=5)
        plt.errorbar(sample_size, hetsc_means_het, yerr=hetsc_stds_het, color='green', capsize=5)
        plt.errorbar(sample_size, hetsc_means_cont, yerr=hetsc_stds_cont, color='black', capsize=5)

        # plot values
        for i in range(len(sample_size)):
            print("sample size: %s, score: %s, std dev: %s" % (sample_size[i], hetsc_means_hom[i], hetsc_stds_hom[i]))

        plt.plot(sample_size, hetsc_exps, color='blue')
        plt.xlabel("Number of simulated individuals")
        plt.ylabel("Heterogeneity Score")

        # label CLiP score
        xloc = 0.8 * (sample_size[-1] - sample_size[0])
        yloc1 = np.interp(xloc, sample_size, hetsc_exps)
        yloc2 = np.interp(xloc, sample_size, hetsc_means_het)
        ax.annotate(s='',xy=(xloc, yloc1), xycoords='data',
				    xytext=(xloc, yloc2),textcoords='data',
				    arrowprops=dict(arrowstyle="<->", color='gray'))
        ax.annotate(s='CLiP Score',xy=(xloc * 1.02, yloc1 + (yloc2-yloc1)*0.75), 
                    xycoords='data',fontsize=10.0,textcoords='data',
                    ha='left', color='gray')

        plt.savefig("simulate_basic_num_inds.eps", format="eps", dpi=1000)
        plt.show()

    else:
        hetsc_means_hom = []
        hetsc_stds_hom = []
        hetsc_means_het = []
        hetsc_stds_het = []
        hetsc_means_cont = []
        hetsc_stds_cont = []
        hetsc_exps = []
        sample_sizes = [1000, 5000, 10000, 20000, 30000, 50000]
        # sample_sizes = [5000, 10000]

        # sample_sizes = [1000, 5000, 10000]
        num_trials = 20
        for sample_size in sample_sizes:
            num_cases = sample_size
            num_conts = sample_size
            num_snps = 100
            h_sq = 0.034
            fixed_ps = np.array([0.2]*num_snps)
            hetscs_hom = []
            hetscs_het = []
            hetscs_cont = []

            # expected score, homogeneous cases
            ps, betas = generate_snp_props(num_snps, fixed_ps, h_sq)
            prev = 0.01
            thresh = norm.ppf(1-prev, loc=0, scale=1)
            hetsc_exp = heterogeneity_expected_corr(ncases = num_cases,
                                                  nconts = num_conts,
                                                  effects = betas,
                                                  thresh = thresh,
                                                  freqs = ps,
                                                  heritability = h_sq, verbose=False)
            hetsc_exps.append(hetsc_exp)

            for i in range(num_trials):
                print("sample_size: %s, trial: %s" % (sample_size, i))

                # homogeneous cases, controls
                cases,conts = generate_homhet_cohort(num_cases, num_conts, num_snps, fixed_ps, h_sq)
                score = heterogeneity(cases,conts)
                hetscs_hom.append(score)

                # heterogeneous cases, controls
                cases,conts = generate_homhet_cohort(num_cases, num_conts, num_snps, fixed_ps, h_sq, het=True)
                score = heterogeneity(cases,conts)
                hetscs_het.append(score)

                # controls, controls
                cases,conts = generate_controls(num_cases, num_conts, num_snps, fixed_ps, h_sq)
                score = heterogeneity(cases,conts)
                hetscs_cont.append(score)

            hetsc_means_hom.append(np.mean(hetscs_hom))
            hetsc_stds_hom.append(np.std(hetscs_hom))
            hetsc_means_het.append(np.mean(hetscs_het))
            hetsc_stds_het.append(np.std(hetscs_het))
            hetsc_means_cont.append(np.mean(hetscs_cont))
            hetsc_stds_cont.append(np.std(hetscs_cont))
        pickle.dump({"hetscs_homs":hetscs_homs,
                     "hetscs_hets":hetscs_hets,
                     "hetscs_conts":hetscs_conts,
                     "hetsc_exps":hetsc_exps,
                     "sample_size":sample_sizes}, open(FILE_PATH, "wb"))
        """
        "hetsc_means_hom":hetsc_means_hom,
        "hetsc_stds_hom":hetsc_stds_hom,
        "hetsc_means_het":hetsc_means_het,
        "hetsc_stds_het":hetsc_stds_het,
        "hetsc_means_cont":hetsc_means_cont,
        "hetsc_stds_cont":hetsc_stds_cont,
        """
if __name__=="__main__":
    run_sample_size()
    run_heritability()
