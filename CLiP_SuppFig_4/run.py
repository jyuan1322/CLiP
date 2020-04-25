import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os.path
import pickle, sys, argparse
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap
sys.path.append('../')
from CLiP import probit_corr, \
                 probit_corr_xy, \
                 expected_corr_unnorm, \
                 heterogeneity, \
                 heterogeneity_expected_corr, \
                 generate_snp_props
                 # generate_cohort, \
                 # generate_homhet_cohort, \
                 # generate_controls

def generate_stratified_controls(num_conts, num_snps, ps_stratif):
    num_sub_pops = len(ps_stratif)
    conts = np.empty(shape=(0, num_snps))
    num_inds = int(num_conts/num_sub_pops)
    for i in range(num_sub_pops):
        ps_sub = ps_stratif[i]
        conts_sub = np.random.binomial(n=2, p=ps_sub, size=(num_inds, num_snps))
        conts = np.concatenate((conts, conts_sub), axis=0)
    return conts

def generate_controls(num_cases, num_conts, num_snps, ps_stratif):
    cases = generate_stratified_controls(num_cases, num_snps, ps_stratif)
    conts = generate_stratified_controls(num_conts, num_snps, ps_stratif)
    return cases,conts

def generate_cohort(num_cases, num_conts, num_snps, betas, ps, ps_stratif, h_sq, thresh):
    # num_snps = len(freqs)
    cases = np.empty((num_cases,num_snps))
    conts = np.empty((num_conts,num_snps))

    # generate controls
    # conts = np.random.binomial(n=2, p=freqs, size=(num_conts,num_snps))
    conts = generate_stratified_controls(num_conts, num_snps, ps_stratif)

    # generate cases
    cur_cases = 0
    subset_size = 10000
    exp_score = 2*np.dot(betas, ps)
    while cur_cases < num_cases:
        # case_samples = np.random.binomial(n=2, p=freqs, size=(subset_size,num_snps))
        case_samples = generate_stratified_controls(subset_size, num_snps, ps_stratif)

        scores = np.dot(case_samples, betas) + \
                 np.random.normal(0,np.sqrt(1-h_sq),subset_size) - \
                 exp_score # set mean score to 0
        """
        # with stratification, the expected score is changed: use the sample mean score instead
        scores = np.dot(case_samples, betas) - \
                 np.mean(np.dot(case_samples, betas)) # set mean score to 0
        # rescale scores to h_sq
        scores = scores / np.std(scores) * np.sqrt(h_sq)
        # add noise to bring to variance=1
        scores = scores + np.random.normal(0, np.sqrt(1-h_sq),subset_size)
        """
        idxs = np.where(scores > thresh)[0]

        if cur_cases+len(idxs) > num_cases:
            idxs = idxs[:(num_cases-cur_cases)]
        cases[cur_cases:cur_cases + len(idxs),:] = case_samples[idxs,:]
        cur_cases += len(idxs)
        # print(cur_cases)
    if cases.shape[0] > num_cases:
        cases = cases[:num_cases, :]
    return cases,conts

def generate_homhet_cohort(num_cases, num_conts, num_snps, ps, ps_stratif, h_sq, het=False):
    # NOTE: ps for homogeneous cases still needed to generate betas
    #num_cases = 5000
    #num_conts = 5000
    #num_snps = 10
    prev = 0.01
    thresh = norm.ppf(1-prev, loc=0, scale=1)
    #h_sq = 0.1

    ps, betas = generate_snp_props(num_snps, ps, h_sq)
    cases, conts = generate_cohort(num_cases=num_cases,
                                   num_conts=num_conts,
                                   # freqs=ps,
                                   num_snps=num_snps, # freqs are sampled randomly when generating stratification
                                   betas=betas,
                                   ps=ps,
                                   ps_stratif=ps_stratif,
                                   h_sq=h_sq,
                                   thresh=thresh)
    if het:
        pi = 0.5
        # conts_temp = np.random.binomial(n=2, p=ps, size=(num_conts,num_snps))
        conts_temp = generate_stratified_controls(num_conts, num_snps, ps_stratif)
        np.random.shuffle(conts_temp)
        cases = np.concatenate((cases[:int(num_cases * pi)], conts_temp[:int(num_cases * (1-pi))]),axis=0)
    return cases,conts


def plot_results(fsts):
    fig, ax = plt.subplots(1,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    cm_g = LinearSegmentedColormap.from_list("green1", ["green","cyan"], N=100)
    cm_r = LinearSegmentedColormap.from_list("red1", ["red",(1,0.9,0)], N=100)
    cm_k = LinearSegmentedColormap.from_list("black1", [(0,0,0),(0.8,0.8,0.8)], N=100)

    colors_green = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0,
                                         vmax=len(fsts)-1), cmap=cm_g)
    colors_green.set_array(range(len(fsts)))
    colors_red = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0,
                                       vmax=len(fsts)-1), cmap=cm_r)
    colors_red.set_array(range(len(fsts)))
    colors_grey = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0,
                                        vmax=len(fsts)-1), cmap=cm_k)
    colors_grey.set_array(range(len(fsts)))

    fig_power, ax_power = plt.subplots(1,1)
    ax_power.spines['right'].set_visible(False)
    ax_power.spines['top'].set_visible(False)
    for i,fst in enumerate(fsts):
        FILE_PATH = "02292020_population_stratif_out_%s.p" % (fst)
        pckl = pickle.load(open(FILE_PATH, "rb"))
        # hetsc_means_hom = pckl["hetsc_means_hom"]
        # hetsc_stds_hom = pckl["hetsc_stds_hom"]
        # hetsc_means_het = pckl["hetsc_means_het"]
        # hetsc_stds_het = pckl["hetsc_stds_het"]
        # hetsc_means_cont = pckl["hetsc_means_cont"]
        # hetsc_stds_cont = pckl["hetsc_stds_cont"]
        hetsc_means_hom = [np.mean(x) for x in pckl["hetscs_homs"]]
        hetsc_stds_hom = [np.std(x) for x in pckl["hetscs_homs"]]
        hetsc_means_het = [np.mean(x) for x in pckl["hetscs_hets"]]
        hetsc_stds_het = [np.std(x) for x in pckl["hetscs_hets"]]
        hetsc_means_cont = [np.mean(x) for x in pckl["hetscs_conts"]]
        hetsc_stds_cont = [np.std(x) for x in pckl["hetscs_conts"]]
        hetsc_exps = pckl["hetsc_exps"]
        h_sqs = pckl["h_sqs"]

        ax.errorbar(h_sqs, hetsc_means_hom, yerr=hetsc_stds_hom, c=colors_red.to_rgba(i), capsize=5)
        ax.errorbar(h_sqs, hetsc_means_het, yerr=hetsc_stds_het, c=colors_green.to_rgba(i), capsize=5)
        ax.errorbar(h_sqs, hetsc_means_cont, yerr=hetsc_stds_cont, c=colors_grey.to_rgba(i), capsize=5)
        ax.plot(h_sqs, hetsc_exps, color='blue')
        ax.set_xlabel("Variance explained by modeled SNPs")
        ax.set_ylabel("Heterogeneity Score")

        """
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
        """
        # power calculation
        power_calc = []
        spcfcty_calc = []
        confd_intvl = norm.ppf(0.95)
        for j,h_sq in enumerate(h_sqs):
            ci_thresh_hom = hetsc_exps[j] + confd_intvl
            hets = pckl["hetscs_hets"][j]
            total_trials = len(hets)
            false_negs = np.sum(hets < ci_thresh_hom)
            power_hom = 1-(false_negs/total_trials)

            fn_hom_thry = norm.cdf(ci_thresh_hom, loc=np.mean(hets), scale=np.std(hets))
            power_hom_thry = 1-(fn_hom_thry/1)
            power_calc.append((power_hom, power_hom_thry))
            # print(power_hom, power_hom_thry)

            # specificity
            homs = pckl["hetscs_homs"][j]
            total_trials_homs = len(hets)
            true_negs = np.sum(homs < ci_thresh_hom)
            spcfcty = true_negs / total_trials_homs

            spcfcty_thry = norm.cdf(ci_thresh_hom, loc=np.mean(homs), scale=np.std(homs))
            spcfcty_calc.append((spcfcty, spcfcty_thry))
            # print(spcfcty, spcfcty_thry)
        ax_power.plot(h_sqs, [x[0] for x in power_calc],
                      c=colors_green.to_rgba(i), linestyle="--")
        # ax_power.plot(h_sqs, [x[1] for x in power_calc], c='g', linestyle=':')
        ax_power.plot(h_sqs, [x[0] for x in spcfcty_calc], c=colors_red.to_rgba(i))
        # ax_power.plot(h_sqs, [x[1] for x in spcfcty_calc], c='r', linestyle=':')
        ax_power.set_xlabel("Variance explained by modeled SNPs")
        ax_power.set_ylabel("Sensitivity/Specificity")
        
    fig.savefig("02292020_population_stratif_plot.eps", format="eps", dpi=500)
    fig_power.savefig("02292020_population_stratif_power.eps", format="eps", dpi=500)

    # make external legend
    fig_leg, ax_leg = plt.subplots(1,1,figsize=(4,7))
    plt.gca().set_visible(False)
    cbar = fig_leg.colorbar(colors_green,ticks=range(len(fsts)),cax=plt.axes([0.1, 0.1, 0.2, 0.8]))
    cbar.ax.set_yticklabels(["" for x in fsts])
    cbar = fig_leg.colorbar(colors_red,ticks=range(len(fsts)),cax=plt.axes([0.3, 0.1, 0.2, 0.8]))
    cbar.ax.set_yticklabels(["" for x in fsts])
    cbar = fig_leg.colorbar(colors_grey,ticks=range(len(fsts)),cax=plt.axes([0.5, 0.1, 0.2, 0.8]))
    cbar.ax.set_yticklabels([str(x) for x in fsts])
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(25)
    plt.savefig("02292020_population_stratif_legend.eps", format="eps", dpi=100)

    plt.show()


def run_heritability():
    # FILE_PATH = "02292020_population_stratif_out.p"
    # pop_allele_diffs = [0.0, 0.05, 0.1, 0.15, 0.2]
    fsts = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
    pop_allele_diffs = 0.5*np.sqrt(fsts)

    # for allele_diff in pop_allele_diffs:
    #     FILE_PATH = "02292020_population_stratif_out_%s.p" % (allele_diff)
    for fst, allele_diff in zip(fsts, pop_allele_diffs):
        FILE_PATH = "02292020_population_stratif_out_%s.p" % (fst)
        if not os.path.exists(FILE_PATH):
            num_snps = 100
            fixed_ps_val = 0.5
            fixed_ps = np.array([fixed_ps_val]*num_snps)
            # generate stratified populations
            """
            ps_stratif = []
            num_sub_pops=4
            for i in range(num_sub_pops):
                ps_sub = np.random.uniform(size=num_snps)
                # ps_sub = [0.2]*num_snps
                ps_stratif.append(ps_sub)
            """

            ps_stratif = [[fixed_ps_val - allele_diff]*int(num_snps/2) + [fixed_ps_val + allele_diff]*int(num_snps/2),
                          [fixed_ps_val + allele_diff]*int(num_snps/2) + [fixed_ps_val - allele_diff]*int(num_snps/2)]

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
            mean_allele_diffs = [] # allele difference between homogeneous cases and controls
            h_sqs = [0.001, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
            # h_sqs = [0.001, 0.025, 0.05, 0.075, 0.1]
            num_trials = 30

            for h_sq in h_sqs:
                num_cases = 30000
                num_conts = 30000
                # num_cases = 5000
                # num_conts = 5000


                hetscs_hom = []
                hetscs_het = []
                hetscs_cont = []
                mean_allele_diff = []

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
                    print("allele_diff: %s, h_sq: %s, trial: %s" % (allele_diff, h_sq, i))

                    # homogeneous cases, controls
                    cases,conts = generate_homhet_cohort(num_cases, num_conts, num_snps, ps, ps_stratif, h_sq)
                    score = heterogeneity(cases,conts)
                    hetscs_hom.append(score)

                    # heterogeneous cases, controls
                    cases,conts = generate_homhet_cohort(num_cases, num_conts, num_snps, ps, ps_stratif, h_sq, het=True)
                    score = heterogeneity(cases,conts)
                    hetscs_het.append(score)
                    mean_allele_diff.append(np.mean(cases) - np.mean(conts))

                    # controls, controls
                    # cases,conts = generate_controls(num_cases, num_conts, num_snps, ps, h_sq)
                    cases, conts = generate_controls(num_cases, num_conts, num_snps, ps_stratif)
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
                mean_allele_diffs.append(mean_allele_diff)
            pickle.dump({"hetscs_homs":hetscs_homs,
                         "hetscs_hets":hetscs_hets,
                         "hetscs_conts":hetscs_conts,
                         "hetsc_exps":hetsc_exps,
                         "mean_allele_diffs":mean_allele_diffs,
                         "h_sqs":h_sqs}, open(FILE_PATH, "wb"))
            """
            "hetsc_means_hom":hetsc_means_hom,
            "hetsc_stds_hom":hetsc_stds_hom,
            "hetsc_means_het":hetsc_means_het,
            "hetsc_stds_het":hetsc_stds_het,
            "hetsc_means_cont":hetsc_means_cont,
            "hetsc_stds_cont":hetsc_stds_cont,
            """
    plot_results(fsts)
if __name__=="__main__":
    run_heritability()
