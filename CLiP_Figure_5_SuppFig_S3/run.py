import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle, sys, argparse
from collections import OrderedDict
from scipy.stats import norm
sys.path.append('../')
from CLiPX_utils import simulate_g2e_unnorm, \
                        get_fixed_expression_params, \
                        generate_cohort
from CLiPX_correlation import expected_corr_unnorm, get_control_cov
from CLiPX import Heterogeneity_TWAS



def plot_heterogeneity(herit, g2wherit=None, log_scale=False):
    corrs = {}
    corrs["het"] = pickle.load(open("score_hetero_%s.p" % (herit), "rb"))
    corrs["hom"] = pickle.load(open("score_homog_%s.p" % (herit), "rb"))
    corrs["cont"] = pickle.load(open("score_cont_%s.p" % (herit), "rb"))
    corrs["exp"] = pickle.load(open("score_pred_%s.p" % (herit), "rb"))
    colors = {"cont":"k", "hom":"r", "het":"g", "exp":"b"}
    fig, ax = plt.subplots(tight_layout=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for c in corrs:
        vals = OrderedDict()
        for a in corrs[c]:
            ncase = a[0]
            score = a[2]
            if ncase not in vals:
                vals[ncase] = []
            vals[ncase].append(score)
        mns = []
        sds = []
        for ncase in vals:
            mns.append(np.mean(vals[ncase]))
            sds.append(np.std(vals[ncase]))
        print(c, mns, sds)
        plt.errorbar([ncase for ncase in vals], mns, yerr=sds, label=c, c=colors[c], capsize=5)

    if log_scale:
        ax.set_xscale('log')
        ax.set_xticks([ncase for ncase in vals])
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        plt.xticks(rotation=315)
        plt.xlabel("log( cohort sample size )")
    else:
        plt.xlabel("cohort sample size")

    plt.ylabel("CLiP-X score")
    if g2wherit is None:
        plt.title(r"$h_E^2$ = {:0.3f}".format(herit))
    else:
        plt.title(r"$V_G^2$ = {:0.3f}, $V_E^2$ = {:0.3f}".format(g2wherit, herit))
    plt.savefig("score_continuous_hetero_%s.eps" % (herit), format="eps", dpi=1000)
    plt.show()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Generate CLiP-X simulations for heterogeneity evaluation')
    parser.add_argument('--load', '-l', dest='load', help="Regenerate plots from existing pickle files", action='store_true')
    parser.add_argument('--g2wherit', dest='in_g2wherit', type=float, help="Desired variance explained generated SNPs on generated expression values. This is fixed across expression variables.", action='store')
    parser.add_argument('--heritability', dest='in_heritability', type=float, help="Desired variance explained by generated expression values on simulated case/control status.", action='store')
    args = parser.parse_args()

    if args.load:
        plot_heterogeneity(0.01, g2wherit=0.1,log_scale=True)
        plot_heterogeneity(0.025, g2wherit=0.1,log_scale=True)
        plot_heterogeneity(0.05, g2wherit=0.1,log_scale=True)
        plot_heterogeneity(0.075, g2wherit=0.1,log_scale=True)
        plot_heterogeneity(0.1, g2wherit=0.1,log_scale=True)
    else:


        num_snps = 100
        num_expr = 10

        freqs, gen2expr_wgtmat, alpha, g2wherit, heritability = get_fixed_expression_params(num_snps, num_expr, args.in_g2wherit, args.in_heritability)

        print("gen2expr_wgtmat:", gen2expr_wgtmat)
        print("alpha:", alpha)
        print("g2wherit:", g2wherit)
        print("herit:", heritability)

        prev = 0.01
        thresh = norm.ppf(1-prev, loc=0, scale=1)
        gene_props = [0.5, 0.5]

        case_size_sets = [100, 250, 500, 750, 1000, 2000]

        num_trials = 2
        cont_res = []
        hom_res = []
        het_res = []
        exp_res = []
        for cs in case_size_sets:
            # correct for negative correlations in homogeneous subgroups
            expcorr, mu_case = expected_corr_unnorm(alpha,
                                    thresh, freqs, gen2expr_wgtmat,
                                    heritability=heritability)
            contcov = get_control_cov(freqs, gen2expr_wgtmat)
            contcorr = np.empty(contcov.shape)
            for i in range(num_expr):
                for j in range(num_expr):
                    contcorr[i,j] = contcov[i,j] / np.sqrt(contcov[i,i] * contcov[j,j])
            mu_cont = np.array([0] * num_expr)
            sd_cont = np.array([1] * num_expr)

            HetScore = Heterogeneity_TWAS()
            HetScore.het_expcorr(cs, cs, expcorr, contcorr, mu_case, mu_cont, sd_cont)

            hetsc = HetScore.get_values(range(num_expr))
            exp_res.append((cs, 0, hetsc))

            # calculate real values
            for nt in range(num_trials):
                # controls
                _,controls1 = simulate_g2e_unnorm(cs, freqs, gen2expr_wgtmat, g2wherit)
                _,controls2 = simulate_g2e_unnorm(cs, freqs, gen2expr_wgtmat, g2wherit)

                HetScore = Heterogeneity_TWAS()
                HetScore.het(controls1, controls2)
                hetsc = HetScore.get_values(range(num_expr))
                cont_res.append((cs, nt, hetsc))

                # homogeneous cases
                case_genos, cases = generate_cohort(num_snps, num_expr, gen2expr_wgtmat,
                                                    freqs, alpha, [1.0], [cs],
                                                    heritability, g2wherit,
                                                    thresh, use_logistic=False)
                control_genos, controls = simulate_g2e_unnorm(cs, freqs, gen2expr_wgtmat, g2wherit)

                print("-"*20)
                HetScore = Heterogeneity_TWAS()
                HetScore.het(cases[0], controls)
                hetsc = HetScore.get_values(range(num_expr))
                hom_res.append((cs, nt, hetsc))

                # heterogeneous cases
                ncase = int(gene_props[0]*cs)
                ncont = int(gene_props[1]*cs)
                # substituted [1.0] for gene_props
                case_genos, cases = generate_cohort(num_snps, num_expr, gen2expr_wgtmat,
                                                    freqs, alpha, [1.0], [ncase],
                                                    heritability,g2wherit,
                                                    thresh, use_logistic=False)
                control_genos, controls = simulate_g2e_unnorm(cs, freqs, gen2expr_wgtmat, g2wherit)
                cont_tmp_genos, cont_tmp = simulate_g2e_unnorm(ncont, freqs, gen2expr_wgtmat, g2wherit)

                HetScore = Heterogeneity_TWAS()
                HetScore.het(cases[0], controls)
                hetsc = HetScore.get_values(range(num_expr))
                print("hetsc homog:", hetsc)

                # add heterogeneity according to gene_props
                cases=cases[0]
                cases = np.concatenate((cases, cont_tmp),axis=0)

                print(np.corrcoef(cases, rowvar=False) -
                      np.corrcoef(controls, rowvar=False))

                HetScore = Heterogeneity_TWAS()
                HetScore.het(cases, controls)
                hetsc = HetScore.get_values(range(num_expr))
                print("hetsc heterog:", hetsc)
                het_res.append((cs, nt, hetsc))
                print("ncases: %s, trial: %s, het score: %s" % (cs, nt, hetsc))

                print("*"*20)
                print(het_res[-1], hom_res[-1], exp_res[-1])
                print("*"*20)
        pickle.dump(cont_res, open("score_cont_%s.p" % (args.in_heritability), "wb"))
        pickle.dump(hom_res, open("score_homog_%s.p" % (args.in_heritability), "wb"))
        pickle.dump(het_res, open("score_hetero_%s.p" % (args.in_heritability), "wb"))
        pickle.dump(exp_res, open("score_pred_%s.p" % (args.in_heritability), "wb"))
