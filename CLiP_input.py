import sys, re, pickle, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
from pprint import pprint
sys.path.append('../')
from CLiP import probit_corr, \
                 probit_corr_xy, \
                 expected_corr_unnorm, \
                 heterogeneity, \
                 heterogeneity_expected_corr

# pip install matplotlib-label-lines
# https://github.com/cphyc/matplotlib-label-lines
from labellines import labelLine, labelLines


def bpflip(snp):
    pair = {"A":"T", "T":"A", "C":"G", "G":"C"}
    return pair[snp]

def getSNPs(SNP_PATH):
    # read SNP file
    # OR and freq pertains to allele 1 (A12)
    snps = {}
    with open(SNP_PATH,"r") as f:
        next(f)
        for line in f:
            # line = line.strip().split(",")
            if not line.startswith("#"):
                line = re.split(''',(?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', line.strip())
                snp_name = line[1]
                if "|" in line[2]: # indel
                    alleles = line[2].split("|")
                else: # snp
                    alleles = [line[2][0], line[2][1]]
                oddsratio = float(line[7].split()[0])
                if oddsratio >= 1:
                    snps[snp_name] = {"eff_allele":alleles[0],
                                      "alt_allele":alleles[1],
                                      "odds_ratio":oddsratio}
                else:
                    snps[snp_name] = {"eff_allele":alleles[1],
                                      "alt_allele":alleles[0],
                                      "odds_ratio":1+(1-oddsratio)}
    return snps

def extract(SNP_PATH, FILE_PATH, SAMPLE_PATH):
    snps = getSNPs(SNP_PATH)

    # extract phenotypes
    phenos = []
    with open(SAMPLE_PATH,"r") as f:
        next(f)
        next(f)
        for line in f:
            line = line.strip().split()
            pheno = int(line[-1])
            phenos.append(pheno)
    phenos = np.array(phenos)

    # extract genotypes
    genos = {}
    with open(FILE_PATH, "r") as f:
        for line in f: # for each SNP
            line = line.split()
            #print(line[:5])
            snp_name = line[1].split(":")
            if len(snp_name[0]) == 1:
                snp_name = snp_name[0] + ":" + snp_name[1]
            else:
                snp_name = snp_name[0]

            if snp_name in snps:
                # A1 A1, A1 A2, A2 A2
                allele1 = line[3]
                allele2 = line[4]

                data = line[5:]
                num_inds = int(len(data)/3)
                geno = []

                for i in range(num_inds):
                    probs = [float(x) for x in data[i*3:i*3+3]]
                    alcnt = np.argmax(probs) # this counts the number of allele2

                    # account for opposite strand
                    if allele1 == snps[snp_name]["eff_allele"] and allele2 == snps[snp_name]["alt_allele"]:
                        alcnt = 2 - alcnt
                    elif allele1 == snps[snp_name]["alt_allele"] and allele2 == snps[snp_name]["eff_allele"]:
                        pass
                    elif bpflip(allele1) == snps[snp_name]["eff_allele"] and bpflip(allele2) == snps[snp_name]["alt_allele"]:
                        alcnt = 2 - alcnt
                    elif bpflip(allele1) == snps[snp_name]["alt_allele"] and bpflip(allele2) == snps[snp_name]["eff_allele"]:
                        pass
                    else:
                        print(snp_name)
                        print(snps[snp_name])
                        print(allele1, allele2)
                        print(line[:10])
                        print(FILE_PATH)
                        print("-"*20)
                        print(allele1, allele2)
                        print(bpflip(allele1), bpflip(allele2))
                        print(snps[snp_name]["eff_allele"], snps[snp_name]["alt_allele"])
                        sys.exit(0)
                    geno.append(alcnt)

                genos[snp_name] = geno
        geno_array = []
        ors = []
        for i in sorted(genos):
            geno_array.append(genos[i])
            ors.append(snps[i]["odds_ratio"])
        ors = np.array(ors)

        genos = np.array(geno_array)
        genos = genos.T

        ctidxs = np.where(phenos == 1)[0]
        csidxs = np.where(phenos == 2)[0]

        conts = genos[ctidxs,:]
        cases = genos[csidxs,:]
        frqs = np.mean(conts, axis=0) / 2

        # remove columns with variance = 0
        keepcols = []
        snpsdel = 0
        for col in range(cases.shape[1]):
            if np.std(cases[:,col]) != 0 and np.std(conts[:,col]) != 0:
                keepcols.append(col)
            else:
                snpsdel += 1
        conts_abr = conts[:, keepcols]
        cases_abr = cases[:, keepcols]
        ors = ors[keepcols]
        frqs = frqs[keepcols]

    # cases_abr has 0-variance SNPs removed
    return cases, conts, cases_abr, conts_abr, snpsdel, ors, frqs

def convertORs(ors, prev):
    betas = []
    prev = 0.01
    thresh = norm.ppf(1-prev, loc=0, scale=1)
    for oddsratio in ors:
        betas.append(norm.ppf(1/(1+np.exp(-(np.log(prev/(1-prev)) + np.log(oddsratio))))) - norm.ppf(0.01))
    betas = np.array(betas)
    return betas


def plot_results(plot_vals, cht_sizes, plot_exp_scores):
    # plot per cohort
    fig, ax = plt.subplots()
    plot_vals = sorted(plot_vals, key=lambda x: x[0])
    ax.scatter([x[0] for x in plot_vals], [x[3] for x in plot_vals], c='k')

    ax.plot(cht_sizes, plot_exp_scores, c='k', label="p = 0.5")
    ax.plot(cht_sizes, plot_exp_scores + norm.ppf(0.95), c='skyblue', label="p = 0.05")
    ax.plot(cht_sizes, plot_exp_scores - norm.ppf(0.95), c='skyblue', label="p = 0.05")
    ax.plot(cht_sizes, plot_exp_scores + norm.ppf(0.99), c='cyan', label="p = 0.01")
    ax.plot(cht_sizes, plot_exp_scores - norm.ppf(0.99), c='cyan', label="p = 0.01")
    ax.plot(cht_sizes, plot_exp_scores + 1, c='cornflowerblue',
                label="p = %0.2f (1 s.d.)" % (1-norm.cdf(1)))
    ax.plot(cht_sizes, plot_exp_scores - 1, c='cornflowerblue',
                label="p = %0.2f (1 s.d.)" % (1-norm.cdf(1)))
    labelLines(plt.gca().get_lines(),xvals=(3000,3000),zorder=2.5)

    ax.set_xlabel("Cases Sample Size")
    ax.set_ylabel("Heterogeneity Score")
    plt.savefig("cohort_score_comparison.png", format="png", dpi=1000)
    plt.show()

if __name__=="__main__":
    PICKLE_OUT="cohort_scores.pickle"

    if os.path.exists(PICKLE_OUT):
        plot_vals, cht_sizes, plot_exp_scores = pickle.load(open(PICKLE_OUT, "rb"))
    else:

        parser = argparse.ArgumentParser()
        parser.add_argument("--snp-path", dest="snp_path", required=True, help="Summary statistic file")
        parser.add_argument("--geno-path", dest="geno_path", required=True, help="Directory of genotype matrices in Oxford .haps format")
        parser.add_argument("--pheno-path", dest="pheno_path", required=True, help="Directory of .sample files containing phenotype labels")
        parser.add_argument("--file-list", dest="file_list", required=True, help="comma-delimited list of cohort names, .haps files, and .sample files, one line per cohort")
        args = parser.parse_args()

        SNP_PATH=args.snp_path
        GENO_PATH=args.geno_path
        PHENO_PATH=args.pheno_path
        FILE_LIST=args.file_list

        files = []

        with open(FILE_LIST, "r") as f:
            for line in f:
                line = line.strip().split(',')
                files.append([line[0], GENO_PATH+"/"+line[1], PHENO_PATH+"/"+line[2]])


        prev = 0.01
        thresh = norm.ppf(1-prev, loc=0, scale=1)
        super_cases = None
        super_conts = None
        super_betas = None
        super_frqs = None
        super_hsq = None
        fisher_p_vals = []
        fisher_sample_sizes = []
        effect_direction = []

        # pickle file for saving correlation tests
        corr_matrices = []

        # for display table and FDR calculation
        table_vals = []

        # for plotting score vs num_cases
        plot_vals = []
        score_std = 1
        for fl in files:
            cht_name, FILE_PATH, SAMPLE_PATH = fl
            try:
                cases, conts, cases_abr, conts_abr, snpsdel, ors, frqs = extract(SNP_PATH,FILE_PATH,SAMPLE_PATH)

                hetsc = heterogeneity(cases_abr, conts_abr)

                # convert odds ratios to liability threshold ratios
                betas = convertORs(ors, prev)
                h_sq = np.sum(np.multiply(np.square(betas), 2*np.multiply(frqs, 1-frqs)))
                # print("h_sq:", h_sq)
                expected_score = heterogeneity_expected_corr(ncases = cases.shape[0],
                                                             nconts = conts.shape[0],
                                                             effects = betas,
                                                             thresh = thresh,
                                                             freqs = frqs,
                                                             heritability = h_sq, verbose=False)
                if snpsdel == 0: # store for combined set
                    super_betas = betas
                    super_frqs = frqs
                    super_hsq = h_sq

                # calculate p-value
                zscore = (hetsc - expected_score)/score_std
                p_val = norm.cdf(-zscore)
                fisher_p_vals.append(p_val)
                fisher_sample_sizes.append(cases.shape[0] + conts.shape[0])
                if zscore >= 0:
                    effect_direction.append(1)
                else:
                    effect_direction.append(-1)

                print("cohort: %s, ncases: %s, nconts: %s, score: %0.2f, ndel: %s, h_sq: %0.4f, exp_score: %0.2f, p-val: %0.2e" % \
                        (cht_name, cases.shape[0], conts.shape[0], hetsc, snpsdel, h_sq, expected_score, p_val))

                table_vals.append([cht_name, cases.shape[0], conts.shape[0], snpsdel, hetsc, expected_score, p_val])
                plot_vals.append((cases.shape[0], cht_name, expected_score, hetsc))
            except Exception as e:
                print(e)
                print(cht_name)

            if super_cases is None:
                super_cases = cases
            else:
                super_cases = np.concatenate((super_cases, cases), axis=0)
            if super_conts is None:
                super_conts = conts
            else:
                super_conts = np.concatenate((super_conts, conts), axis=0)
        super_hetsc = heterogeneity(super_cases, super_conts)

        # calculate expected score for the entire cohort
        super_exp_score = heterogeneity_expected_corr(ncases = super_cases.shape[0],
                                                      nconts = super_conts.shape[0],
                                                      effects = super_betas,
                                                      thresh = thresh,
                                                      freqs = super_frqs,
                                                      heritability = super_hsq, verbose=False)
        zscore = (super_hetsc - super_exp_score)/score_std
        p_val = norm.cdf(-zscore)
        print("total cases: %s, total conts: %s, total score: %0.2f, exp_score: %0.2f, p-val: %0.2e" % (super_cases.shape[0], super_conts.shape[0], super_hetsc, super_exp_score, p_val))

        # calculate sum of expected scores
        sum_score = np.sum([x[4] for x in table_vals])
        exp_sum_score = np.sum([x[5] for x in table_vals])
        sum_std = np.sqrt(len(table_vals))
        zscore = (sum_score - exp_sum_score) / sum_std
        pval = norm.cdf(-zscore)
        print("sum score: %s, exp sum score: %s, sum std: %s, z-score: %s, p-val: %s" % (sum_score, exp_sum_score, sum_std, zscore, pval))

        # calculate Fisher's p-value
        fisher_p_vals = np.array(fisher_p_vals)
        fisher_sample_sizes = np.sqrt(np.array(fisher_sample_sizes)) # NOTE: the w_i's are square rooted
        efect_direction = np.array(effect_direction)
        chisq = -2 * np.sum(np.log(fisher_p_vals))
        chi2_pval = 1 - chi2.cdf(chisq, df=2*len(fisher_p_vals))
        print("Fisher chi2 p-value: %s" % (chi2_pval))

        # calculate meta-analysis Z-score
        zscores = np.multiply(norm.cdf(1 - fisher_p_vals), effect_direction) # one-sided test
        print("zscores:", zscores)
        denom = np.sqrt(np.sum(np.square(fisher_sample_sizes)))
        meta_Z = np.dot(zscores, fisher_sample_sizes) / denom
        print("meta_Z:", meta_Z)


        # table and FDR rate
        print("\n"*10)
        table_vals.sort(key=lambda x: x[-1])
        num_chts = len(table_vals)
        FDR_rate = 0.3333
        for i,tv in enumerate(table_vals, 1): # start at 1
            bh_val = float(i)/num_chts * FDR_rate
            tv.append(bh_val)
            print("%s & %s/%s & %s & %0.2f & %0.2f & %0.3f & %0.3f" % tuple(tv))


        # smooth curve of expectations
        cht_sizes = range(100, 4000, 250)
        plot_exp_scores = []
        for cs in cht_sizes:
            expected_score = heterogeneity_expected_corr(ncases = cs,
                                                         nconts = cs,
                                                         effects = super_betas,
                                                         thresh = thresh,
                                                         freqs = super_frqs,
                                                         heritability = super_hsq, verbose=False)
            plot_exp_scores.append(expected_score)
        plot_exp_scores = np.array(plot_exp_scores)
        pickle.dump( (plot_vals, cht_sizes, plot_exp_scores), open(PICKLE_OUT, "wb"))
    plot_results(plot_vals, cht_sizes, plot_exp_scores)

