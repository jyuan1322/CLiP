import matplotlib.pyplot as plt
import numpy as np
import pickle, sys, os, argparse
from scipy import integrate
from scipy.stats import norm
from scipy.special import gamma
from scipy.special import hyp2f1
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from pprint import pprint
import matplotlib as mpl
sys.path.append('../')
from CLiPX import Heterogeneity_GWAS
from CLiP_input import getSNPs


# set font
font = {'size':14}
mpl.rc('font', **font)



def simulate_individual(num, freqs):
    num = int(num)
    inds = np.empty([num, len(freqs)])
    for i, p in enumerate(freqs):
        sprobs = [(1-p)*(1-p), 2*p*(1-p), p*p]
        inds[:, i] = np.random.choice(3,size=num,p=sprobs)
    return inds

def generate_cohort_logit(ORs, freqs, thresh, case_sizes, control_size):
    """
    case_sizes should be a list, i.e. [5000,5000]
    """
    controls = simulate_individual(control_size, freqs)

    cases = []
    for c in range(len(case_sizes)):
        cases.append(np.empty((case_sizes[c], len(freqs))))

    all_count = 0 # total cases
    cur_count = 0
    cur_case_size = [0] * len(case_sizes)
    for case_size in case_sizes:
        all_count += case_size

    batch = 10000
    beta_logits = np.log(ORs)
    exp_score = 2*np.dot(beta_logits.flatten(), freqs)
    while cur_count < all_count:
        indivs = simulate_individual(batch, freqs)
        scores = 1/(1 + np.exp(-(np.dot(indivs, beta_logits) - exp_score + thresh)))
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                if np.random.rand() < scores[i,j] and cur_case_size[j] < case_sizes[j]:
                    cases[j][cur_case_size[j], :] = indivs[i,:]
                    cur_count += 1
                    cur_case_size[j] += 1
        print("[CASE] generated", cur_count, "out of", all_count)
    return cases,controls

def generate_cohort_risch(ORs, freqs, prev, case_sizes, control_size):
    """
    case_sizes should be a list, i.e. [5000,5000]
    """
    controls = simulate_individual(control_size, freqs)

    cases = []
    for c in range(len(case_sizes)):
        cases.append(np.empty((case_sizes[c], len(freqs))))

    all_count = 0 # total cases
    cur_count = 0
    cur_case_size = [0] * len(case_sizes)
    for case_size in case_sizes:
        all_count += case_size

    # batch = 3000
    batch = 10000
    # Risch constant specifying the prevalence
    A = []
    for j in range(len(case_sizes)):
        # an approximation; these values should be set to constants anyway
        or_mean = np.mean(ORs)
        freqs_mean = np.mean(freqs)
        num_snps = len(freqs)

        # mu_snps = 2*num_snps*freqs_mean
        # sigma_snps = np.sqrt(2*num_snps*(freqs_mean)*(1-freqs_mean))
        # exp_risch = or_mean**(mu_snps) + np.exp(0.5 * np.log(or_mean)**2 * (sigma_snps)**2)
        # A.append(prev / np.product(np.power(ORs[:,j], 2*freqs)))
        # A.append(prev / exp_risch)
        mu_snps = 2*num_snps*freqs_mean
        sigma_snps = np.sqrt(2*num_snps*(freqs_mean)*(1-freqs_mean))
        exp_val = np.exp((-mu_snps**2 + (sigma_snps**2 * np.log(or_mean) + mu_snps)**2)/(2*sigma_snps**2))
        A.append(prev / exp_val)

    while cur_count < all_count:
        indivs = simulate_individual(batch, freqs)
        prevalence_test = []
        for i in range(indivs.shape[0]):
            for j in range(len(case_sizes)):
                # prob = prev * np.product(np.power(ORs[:,j], indivs[i,:]))
                prob = A[j] * np.product(np.power(ORs[:,j], indivs[i,:]))                

                # automatically make a case if prob > 1
                if (prob > 1 or np.random.binomial(n=1,p=prob) == 1) and cur_case_size[j] < case_sizes[j]:
                    cases[j][cur_case_size[j], :] = indivs[i,:]
                    cur_count += 1
                    cur_case_size[j] += 1
                    prevalence_test.append(1)
                else:
                    prevalence_test.append(0)
        print("[CASE] generated", cur_count, "out of", all_count, "prevalence %0.5f" % (np.sum(prevalence_test)/len(prevalence_test)))
    return cases,controls

def hetscore_numinds(ORs, freqs, prev, case_sizes, control_size, snp_idxs, hetero, use_logistic, num_trials=3):

    hetscores=[]
    for nt in range(num_trials):
        print("hetero: %s, use_logistic: %s, case sizes: %s, trial: %s" % (hetero, use_logistic, case_sizes, nt))
        if use_logistic:
            thresh = -np.log(1/prev - 1)
            cases, controls = generate_cohort_logit(ORs, freqs, thresh,
                                                   case_sizes, control_size)
        else:
            cases, controls = generate_cohort_risch(ORs, freqs, prev,
                                                   case_sizes, control_size)

        # cases are a list of genotype matrices by subtype
        cases_all = cases[0]
        for i in range(1,len(cases)):
            cases_all = np.concatenate((cases_all, cases[i]),axis=0)
        if hetero: 
            case_l = cases_all.shape[0]
            cont2 = simulate_individual(int(case_l/2), freqs)
            cases_all = np.concatenate((cases_all[:int(case_l/2), :], cont2), axis=0)
        
        HetScore = Heterogeneity_GWAS()
        HetScore.het(cases_all, controls)
        for si in snp_idxs:
            hetsc = HetScore.get_values(si)
            # print("het score:", hetsc)
            hetscores.append(hetsc)
    return np.mean(hetscores), np.std(hetscores)

def load_results():
    # load pickled results
    num_trials = 20
    ax = plt.subplot(111)

    expmt = "high"
    # expmt = "low"

    cohort_sizes = pickle.load(open("cohort_sizes_%s_or.p" % (expmt), "rb"))

    temp1 = pickle.load(open("simulate_hom_lt_%s_or.p" % (expmt), "rb"))
    mean_res_hom_lts = temp1[0]
    std_res_hom_lts = temp1[1]
    ax.errorbar(cohort_sizes, mean_res_hom_lts, yerr=std_res_hom_lts,
                label="Homog cases, Logistic", color="#9b9bff", capsize=5) # light blue

    temp2 = pickle.load(open("simulate_het_lt_%s_or.p" % (expmt), "rb"))
    mean_res_het_lts = temp2[0]
    std_res_het_lts = temp2[1]
    ax.errorbar(cohort_sizes, mean_res_het_lts, yerr=std_res_het_lts,
                label="Heterog cases, Logistic", color="blue", capsize=5)

    temp = pickle.load(open("simulate_cont_%s_or.p" % (expmt), "rb"))
    mean_res_cont = temp[0]
    std_res_cont = temp[1]
    ax.errorbar(cohort_sizes, mean_res_cont, yerr=std_res_cont,
                label="control", color='k', capsize=5)

    temp3 = pickle.load(open("simulate_hom_rs_%s_or.p" % (expmt), "rb"))
    mean_res_hom_rss = temp3[0]
    std_res_hom_rss = temp3[1]
    ax.errorbar(cohort_sizes, mean_res_hom_rss, yerr=std_res_hom_rss, 
                label="Homog cases, Risch", color="#ff8c8c", capsize=5) # light red

    temp4 = pickle.load(open("simulate_het_rs_%s_or.p" % (expmt), "rb"))
    mean_res_het_rss = temp4[0]
    std_res_het_rss = temp4[1]
    ax.errorbar(cohort_sizes, mean_res_het_rss, yerr=std_res_het_rss, 
                label="Heterog cases, Risch", color="red", capsize=5)

    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Number of cases and controls")
    ax.set_ylabel("Heterogeneity Score")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.25,
                     box.width, box.height * 0.75])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    plt.savefig("simulate_neg_cor_test_%s_or.eps" % (expmt), format="eps", dpi=1000)
    plt.show()


if __name__=="__main__":

    """
    # verify prevalence
    snp = np.random.binomial(n=2, p=0.5, size=100000)
    OR = 1.2
    risks = np.power(OR, snp)
    print(np.mean(risks))
    mu = 2 * 0.5
    sigma = np.sqrt(2 * 0.5 * (1-0.5))
    print(np.exp( (-mu**2 + (sigma**2 * np.log(OR) + mu)**2)/(2*sigma**2) ))
    sys.exit(0)
    """



    FILE_PATH="cohort_sizes.p"
    if os.path.exists(FILE_PATH):
        load_results()
    else:

        parser = argparse.ArgumentParser()
        parser.add_argument("--snp-path", dest="snp_path", required=False, default=None, help="Summary statistic file")
        args = parser.parse_args()

        # num_trials = 10
        # cohort_sizes = [1000, 5000, 10000, 20000, 30000, 50000, 75000, 100000]
        num_trials = 3
        cohort_sizes = [50000, 75000, 100000]
        pickle.dump(cohort_sizes, open("cohort_sizes.p", "wb"))


        if args.snp_path is None:

            num_snps = 100
            freqs = np.array([0.2] * num_snps)
            ORs = np.array([[1.2] * num_snps]).T
            prev = 0.01

            """
            num_snps = 100
            freqs = np.array([0.2] * num_snps)
            ORs = np.array([[1.06] * num_snps]).T
            prev = 0.01
            """
        else:
            # run on input SNP list
            snps = getSNPs(args.snp_path)
            prev = 0.01
            num_snps = len(snps)
            snp_list = [(snps[s]["odds_ratio"],
                         snps[s]["eff_freq"]) for s in snps]
            snp_sel = sorted(snp_list, key=lambda x:x[0], reverse=True)

            freqs = np.array([x[1] for x in snp_sel])
            ORs = np.array([[x[0] for x in snp_sel]]).T

        # control
        mean_res_conts = []
        std_res_conts = []
        # snp_idxs = [range(0,int(len(freqs)/2)), range( int(len(freqs)/2), len(freqs))]
        snp_idxs = [range(len(freqs))]
        for c in cohort_sizes:
            case_sizes = [c]
            control_size = c
            hetscores = []
            for i in range(num_trials):
                controls1 = simulate_individual(control_size, freqs)
                controls2 = simulate_individual(control_size, freqs)

                HetScore = Heterogeneity_GWAS()
                HetScore.het(controls1, controls2)

                for si in snp_idxs:
                    hetsc = HetScore.get_values(si)
                    hetscores.append(hetsc)

            mean_res_conts.append(np.mean(hetscores))
            std_res_conts.append(np.std(hetscores))

        print("mean_res_conts:", mean_res_conts)
        print("std_res_conts:", std_res_conts)
        pickle.dump((mean_res_conts, std_res_conts), open("simulate_cont.p", "wb"))

        # homogeneous case Risch
        # ORs = np.array([[1.16] * 10]).T
        mean_res_hom_rss = []
        std_res_hom_rss = []
        # snp_idxs = [range(0,int(len(freqs)/2)), range( int(len(freqs)/2), len(freqs))]
        snp_idxs = [range(len(freqs))]
        for c in cohort_sizes:
            case_sizes = [c]
            control_size = c
            mean_res_hom_rs, std_res_hom_rs = hetscore_numinds(ORs, freqs, prev, case_sizes, control_size, snp_idxs, hetero=False, use_logistic=False, num_trials=num_trials)
            mean_res_hom_rss.append(mean_res_hom_rs)
            std_res_hom_rss.append(std_res_hom_rs)
        pickle.dump((mean_res_hom_rss, std_res_hom_rss), open("simulate_hom_rs.p", "wb"))

        
        # hetero case Risch
        # ORs = np.array([[1.16] * 10]).T
        mean_res_het_rss = []
        std_res_het_rss = []
        # snp_idxs = [range(0,int(len(freqs)/2)), range( int(len(freqs)/2), len(freqs))]
        snp_idxs = [range(len(freqs))]
        for c in cohort_sizes:
            case_sizes = [c]
            control_size = c
            mean_res_het_rs, std_res_het_rs = hetscore_numinds(ORs, freqs, prev, case_sizes, control_size, snp_idxs, hetero=True, use_logistic=False, num_trials=num_trials)
            mean_res_het_rss.append(mean_res_het_rs)
            std_res_het_rss.append(std_res_het_rs)
        pickle.dump((mean_res_het_rss, std_res_het_rss), open("simulate_het_rs.p", "wb"))


        # homogeneous case logit
        # ORs = np.array([[1.16] * 10]).T
        mean_res_hom_lts = []
        std_res_hom_lts = []
        # snp_idxs = [range(0,int(len(freqs)/2)), range( int(len(freqs)/2), len(freqs))]
        snp_idxs = [range(len(freqs))]
        for c in cohort_sizes:
            case_sizes = [c]
            control_size = c
            mean_res_hom_lt, std_res_hom_lt = hetscore_numinds(ORs, freqs, prev, case_sizes, control_size, snp_idxs, hetero=False, use_logistic=True, num_trials=num_trials)
            mean_res_hom_lts.append(mean_res_hom_lt)
            std_res_hom_lts.append(std_res_hom_lt)
        pickle.dump((mean_res_hom_lts, std_res_hom_lts), open("simulate_hom_lt.p", "wb"))

        # hetero case logit
        # ORs = np.array([[1.16] * 10]).T
        mean_res_het_lts = []
        std_res_het_lts = []
        #snp_idxs = [range(0,int(len(freqs)/2)), range( int(len(freqs)/2), len(freqs))]
        snp_idxs = [range(len(freqs))]
        for c in cohort_sizes:
            case_sizes = [c]
            control_size = c
            mean_res_het_lt, std_res_het_lt = hetscore_numinds(ORs, freqs, prev, case_sizes, control_size, snp_idxs, hetero=True, use_logistic=True, num_trials=num_trials)
            mean_res_het_lts.append(mean_res_het_lt)
            std_res_het_lts.append(std_res_het_lt)
        pickle.dump((mean_res_het_lts, std_res_het_lts), open("simulate_het_lt.p", "wb"))
        
        plt.errorbar(cohort_sizes, mean_res_conts, yerr=std_res_conts, label="cont")
        plt.errorbar(cohort_sizes, mean_res_hom_rss, yerr=std_res_hom_rss, label="hom_rs")
        plt.errorbar(cohort_sizes, mean_res_het_rss, yerr=std_res_het_rss, label="het_rs")
        plt.errorbar(cohort_sizes, mean_res_hom_lts, yerr=std_res_hom_lts, label="hom_lt")
        plt.errorbar(cohort_sizes, mean_res_het_lts, yerr=std_res_het_lts, label="het_lt")
        plt.legend()
        plt.show()
