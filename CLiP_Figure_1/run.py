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
sys.path.append('../')
from CLiPX import Heterogeneity_GWAS
from CLiP_input import getSNPs
from CLiP import generate_snp_props, heterogeneity_expected_corr
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_corr(x, effect, freq, beta0):
    ex = 2*freq
    res = sigmoid(beta0 + effect*(x-ex))
    return res

def sigmoid_corr_xy(x,y, efi, efj, freqi, freqj, beta0):
    ex = 2*freqi
    ey = 2*freqj
    res = sigmoid(beta0 + efi*(x-ex) + efj*(y-ey))
    return res

def expected_corr_unnorm_logistic(ORs, freqs, prev, verbose=False):

    num_snps = len(ORs)
    effects = np.log(ORs)
    beta0 = -np.log(1/prev - 1)
    # calculate E[X], E[X^2]
    ex = []
    ex2 = []
    for i in range(num_snps):
        margx = []
        for x in range(3):
            m = sigmoid_corr(x, effects[i], freqs[i], beta0) * freqs[i]**x * (1-freqs[i])**(1-x)
            if x==1:
                m *= 2
            margx.append(m)
        margx = np.array(margx) / np.sum(margx)
        ex.append(margx[1]*1 + margx[2]*2)
        ex2.append(margx[1]*1 + margx[2]*4)

    # calculate E[XY]
    exy = np.empty((num_snps,num_snps))
    for i in range(num_snps):
        for j in range(i+1,num_snps):
            efi = effects[i]
            efj = effects[j]
            freqi = freqs[i]
            freqj = freqs[j]

            margxy = np.empty((3,3))
            for x in range(3):
                for y in range(3):
                    margxy[x,y] = sigmoid_corr_xy(x,y, efi, efj, freqi, freqj, beta0) * freqi**x * (1-freqi)**(1-x) * freqj**y * (1-freqj)**(1-y)
                    if x==1:
                        margxy[x,y] *= 2
                    if y==1:
                        margxy[x,y] *= 2
            margxy /= np.sum(margxy)
            exy[i,j] = (margxy[1,1] + 2*margxy[1,2] + 2*margxy[2,1] + 4*margxy[2,2])


    # calculate expected correlation
    rho = np.empty((num_snps,num_snps))
    for i in range(num_snps):
        rho[i,i] = 1.0
        for j in range(i+1,num_snps):
            rho[i,j] = (exy[i,j] - ex[i] * ex[j]) / np.sqrt(ex2[i] - ex[i]**2) / np.sqrt(ex2[j] - ex[j]**2)
            rho[j,i] = rho[i,j]
    return rho, ex

def heterogeneity_expected_corr_logit(ncases, nconts, ORs, freqs, prev, verbose=False):
    num_snps = len(ORs)
    N = ncases
    Np = nconts

    R, ex = expected_corr_unnorm_logistic(ORs, freqs, prev, verbose=False)
    Rp = np.eye(num_snps)
    Y = np.sqrt(N*Np/(N+Np)) * (R-Rp)

    z = np.zeros(num_snps)
    for i in range(num_snps):
        p_case = ex[i]/2
        p_control = freqs[i]
        gamma = p_case/(1-p_case) / (p_control/(1-p_control))
        z[i] = np.sqrt(p_control*(1-p_control)) * (gamma-1) / (p_control*(gamma-1) + 1)


    numer = 0.0
    denom = 0.0
    for i in range(num_snps):
        for j in range(i+1,num_snps):
            wij = z[i] * z[j]
            yij = Y[i,j]
            numer += wij * yij
            denom += wij * wij
    score = numer / np.sqrt(denom)
    return score











def convertORs(ors, prev):
    betas = []
    prev = 0.01
    thresh = norm.ppf(1-prev, loc=0, scale=1)
    for oddsratio in ors:
        betas.append(norm.ppf(1/(1+np.exp(-(np.log(prev/(1-prev)) + np.log(oddsratio))))) - norm.ppf(prev))
    betas = np.array(betas)
    return betas

def generate_cohort(num_cases,num_conts,freqs,betas,h_sq,thresh):
    num_snps = len(freqs)
    cases = np.empty((num_cases,num_snps))
    conts = np.empty((num_conts,num_snps))

    # generate controls
    conts = np.random.binomial(n=2, p=freqs, size=(num_conts,num_snps))

    # generate cases
    cur_cases = 0
    subset_size = 10000
    exp_score = 2*np.dot(betas, freqs)
    while cur_cases < num_cases:
        case_samples = np.random.binomial(n=2, p=freqs, size=(subset_size,num_snps))
        scores = np.dot(case_samples, betas) + \
                 np.random.normal(0,np.sqrt(1-h_sq),subset_size) - \
                 exp_score # set mean score to 0
        idxs = np.where(scores > thresh)[0]

        if cur_cases+len(idxs) > num_cases:
            idxs = idxs[:(num_cases-cur_cases)]
        cases[cur_cases:cur_cases + len(idxs),:] = case_samples[idxs,:]
        cur_cases += len(idxs)
        # print("liability:" + str(cur_cases))
    if cases.shape[0] > num_cases:
        cases = cases[:num_cases, :]
    return cases,conts

def generate_cohort_logistic(num_cases, num_conts, freqs, ORs, prev):
    const = -np.log(1/prev - 1)
    beta_logits = np.log(ORs)
    num_snps = len(freqs)
    cases = np.empty((num_cases,num_snps))

    # generate controls
    conts = np.random.binomial(n=2, p=freqs, size=(num_conts, num_snps))

    # generate cases
    cur_cases = 0
    subset_size = 10000
    exp_score = 2*np.dot(beta_logits, freqs)
    while cur_cases < num_cases:
        case_samples = np.random.binomial(n=2, p=freqs, size=(subset_size,num_snps))
        scores = np.dot(case_samples, beta_logits) + \
                 const - \
                 exp_score # set mean score to 0
        scores = 1.0/(1.0+np.exp(-scores))
        case_labels = np.random.binomial(n=1, p=scores, size=subset_size)
        idxs = np.where(case_labels==1)[0]

        if cur_cases+len(idxs) > num_cases:
            idxs = idxs[:(num_cases-cur_cases)]
        cases[cur_cases:cur_cases + len(idxs),:] = case_samples[idxs,:]
        cur_cases += len(idxs)
        # print("logit: " + str(cur_cases))
    if cases.shape[0] > num_cases:
        cases = cases[:num_cases, :]
    return cases,conts

def run():
    """
    num_snps = 100
    freqs = np.array([0.5] * num_snps)
    ORs = np.array([[1.06] * num_snps]).T
    prev = 0.01

    hetsc = heterogeneity_expected_corr_logit(ncases=10000, nconts=10000, ORs=ORs, freqs=freqs, prev=prev, verbose=False)
    print(hetsc)
    """



    # tests start here
    FILE_PATH="logistic_vs_liability.pickle"
    num_snps = 10
    if os.path.exists(FILE_PATH):
        results = pickle.load(open(FILE_PATH, "rb"))
        ORlist = results["ORlist"]
        liab_var_exps = results["liab_var_exps"]
        all_pred_scores_logit = results["all_pred_scores_logit"]
        all_pred_scores_liab = results["all_pred_scores_liab"]
        all_logit_scores = results["all_logit_scores"]
        all_liab_scores = results["all_liab_scores"]
        all_logit_allele_count_diffs = results["all_logit_allele_count_diffs"]
        all_liab_allele_count_diffs = results["all_liab_allele_count_diffs"]
        # TODO: write num_snps to file
    else:
        num_indivs = 10000
        all_logit_scores = []
        all_liab_scores = []
        all_pred_scores_logit = []
        all_pred_scores_liab = []
        all_logit_allele_count_diffs = []
        all_liab_allele_count_diffs = []
        liab_var_exps = []

        freqs = np.array([0.5] * num_snps)
        prev = 0.01
        ORlist = [1.05, 1.10, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6]
        # ORlist = [1.2, 1.3]
        num_trials = 10
        for i in range(len(ORlist)):
            ORs = np.array([[ORlist[i]] * num_snps]).T
            
            pred_hetsc_logit = heterogeneity_expected_corr_logit(ncases=num_indivs,
                                                                 nconts=num_indivs,
                                                                 ORs=ORs,
                                                                 freqs=freqs,
                                                                 prev=prev,
                                                                 verbose=False)
            all_pred_scores_logit.append(pred_hetsc_logit)
            
            thresh = norm.ppf(1-prev, loc=0, scale=1)
            betas = convertORs(ORs, prev)
            liability_varexp = np.dot(np.square(betas.flatten()),
                                      2*np.multiply(freqs, 1-freqs))
            liab_var_exps.append(liability_varexp)
            pred_hetsc_liab = heterogeneity_expected_corr(ncases=num_indivs,
                                                          nconts=num_indivs,
                                                          effects=betas.flatten(),
                                                          thresh=thresh,
                                                          freqs=freqs,
                                                          heritability=liability_varexp,
                                                          verbose=False)
            all_pred_scores_liab.append(pred_hetsc_liab)

            logit_scores = []
            liab_scores = []
            logit_allele_count_diffs = []
            liab_allele_count_diffs = []
            for nt in range(num_trials):
                print("OR: %s, trial: %s" % (ORlist[i], nt)) 
                cases_logit, conts_logit = generate_cohort_logistic(num_cases=num_indivs,
                                                                num_conts=num_indivs,
                                                                freqs=freqs,
                                                                ORs=ORs.flatten(),
                                                                prev=prev)

                cases_liab, conts_liab = generate_cohort(num_cases=num_indivs,
                                                         num_conts=num_indivs,
                                                         freqs=freqs,
                                                         betas=betas.flatten(),
                                                         h_sq=liability_varexp,
                                                         thresh=thresh)
                # test heterogeneity
                HetScore_logit = Heterogeneity_GWAS()
                HetScore_logit.het(cases_logit, conts_logit)
                hetsc_logit = HetScore_logit.get_values(range(num_snps))
                # equivalent: heterogeneity(cases_logit, conts_logit), from CLiP import heterogeneity
                HetScore_liab = Heterogeneity_GWAS()
                HetScore_liab.het(cases_liab, conts_liab)
                hetsc_liab = HetScore_liab.get_values(range(num_snps))
                
                # print("logit:", hetsc_logit)
                # print(np.mean(np.corrcoef(cases_logit, rowvar=False)[np.triu_indices(num_snps, k=1)]))
                # print(HetScore_logit.get_weights())
                logit_scores.append(hetsc_logit)
                liab_scores.append(hetsc_liab)
                logit_allele_count_diffs.append(np.mean(cases_logit) - np.mean(conts_logit))
                liab_allele_count_diffs.append(np.mean(cases_liab) - np.mean(conts_liab))
            all_logit_scores.append(logit_scores)
            all_liab_scores.append(liab_scores)
            all_logit_allele_count_diffs.append(logit_allele_count_diffs)
            all_liab_allele_count_diffs.append(liab_allele_count_diffs)


        pickle.dump({"ORlist":ORlist,
                     "liab_var_exps":liab_var_exps,
                     "all_pred_scores_logit":all_pred_scores_logit,
                     "all_pred_scores_liab":all_pred_scores_liab,
                     "all_logit_scores":all_logit_scores,
                     "all_liab_scores":all_liab_scores,
                     "all_logit_allele_count_diffs":all_logit_allele_count_diffs,
                     "all_liab_allele_count_diffs":all_liab_allele_count_diffs}, open(FILE_PATH, "wb"))

    fig, ax = plt.subplots()
    plt.plot(ORlist, all_pred_scores_logit, linestyle='--', color='k')
    mean_logit_scores = [np.mean(scs) for scs in all_logit_scores]
    std_logit_scores = [np.std(scs) for scs in all_logit_scores]
    plt.errorbar(ORlist, mean_logit_scores, yerr=std_logit_scores, capsize=5, alpha=0.8, color='cyan')
    plt.plot(ORlist, all_pred_scores_liab, linestyle='--', color='k')
    mean_liab_scores = [np.mean(scs) for scs in all_liab_scores]
    std_liab_scores = [np.std(scs) for scs in all_liab_scores]
    plt.errorbar(ORlist, mean_liab_scores, yerr=std_liab_scores, capsize=5, alpha=0.8, color='b')

    liab_xaxis_labels = ["(%.4f)" % (i/num_snps) for i in liab_var_exps]
    ignore_labels = [1,3,5]
    for i in ignore_labels:
        liab_xaxis_labels[i] = ""

    ax.set_xticks(ORlist)
    ax.set_xticklabels(["%s\n%s" % (i[0],i[1]) for i in zip(ORlist, liab_xaxis_labels)])
    ax.set_xlabel("SNP Odds Ratio (per-SNP variance explained)")
    ax.set_ylabel("Heterogeneity Score")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("logistic_vs_liability_het_scores.eps", format="eps", dpi=500)

    fig, ax = plt.subplots()
    mean_logit_allele_count_diffs = [np.mean(diffs) for diffs in all_logit_allele_count_diffs]
    std_logit_allele_count_diffs = [np.std(diffs) for diffs in all_logit_allele_count_diffs]
    mean_liab_allele_count_diffs = [np.mean(diffs) for diffs in all_liab_allele_count_diffs]
    std_liab_allele_count_diffs = [np.std(diffs) for diffs in all_liab_allele_count_diffs]
    plt.errorbar(ORlist, mean_logit_allele_count_diffs, yerr=std_logit_allele_count_diffs, capsize=5, color='cyan')
    plt.errorbar(ORlist, mean_liab_allele_count_diffs, yerr=std_liab_allele_count_diffs, capsize=5, color='b')

    ax.set_xticks(ORlist)
    ax.set_xticklabels(["%s\n%s" % (i[0],i[1]) for i in zip(ORlist, liab_xaxis_labels)])
    ax.set_xlabel("SNP Odds Ratio (per-SNP variance explained)")
    ax.set_ylabel("Difference in mean allele count (Cases - Controls)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("logistic_vs_liability_casecont_allele_diffs.eps", format="eps", dpi=500)

    fig, ax = plt.subplots()
    x = np.linspace(-5, 5, 500)
    y_logit = 1/(1+np.exp(-x)) * (1 - 1/(1+np.exp(-x)))
    # variance of sigmoid is pi^2/3, so divide by square root of this
    # x_scaled = x / (np.pi/np.sqrt(3))
    # y_logit_scaled = 1/(1+np.exp(-x_scaled)) * (1 - 1/(1+np.exp(-x_scaled)))
    y_liab = norm.pdf(x)
    plt.plot(x,y_logit, color='cyan')
    # plt.plot(x,y_logit_scaled, color='cyan', linestyle=':')
    plt.plot(x,y_liab, color='b')
    plt.savefig("logistic_vs_liability_pdfs.eps", format="eps", dpi=500)
    plt.show()



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
    run()
