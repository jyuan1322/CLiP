import os.path
import numpy as np
import sys
from scipy import integrate
from scipy.stats import norm
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pprint import pprint
sys.path.append('../')
from CLiPY_utils import generate_pss_model_simple, \
                  calc_var_from_geno, \
                  generate_population
from CLiPY import heterogeneity, \
                 log1, log1p5, log3, \
                 step, sigmoid, linear, \
                 polynom2, polynom4, polynom6, \
                 run_heterogeneity_on_pop, \
                 run_cont_heterogeneity_on_pop, \
                 get_exp_heterogeneity


def simulate_mult_n(file_path, runs=1, num_snps=100, h=0.3, num_inds_vals=(1000, 5000, 10000, 20000, 50000, 100000), bzs=[-0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5]):
    independent_snps = generate_pss_model_simple(num_snps, h)
    hetero_snps = generate_pss_model_simple(num_snps, h)


    weight_funcs = {"log1":log1, "log1p5":log1p5, "log3":log3, "sigmoid":sigmoid, "linear":linear, "step":step, "polynom2":polynom2, "polynom4":polynom4, "polynom6":polynom6}
    hom_thresh_runs = {}
    het_thresh_runs = {}
    hom_continuous_runs = {}
    het_continuous_runs = {}
    hom_contin_exp_runs = {}

    for bz in bzs:
        hom_thresh_runs[bz] = {ni:[] for ni in num_inds_vals}
        het_thresh_runs[bz] = {ni:[] for ni in num_inds_vals}
    for ws in weight_funcs:
        hom_continuous_runs[ws] = {ni:[] for ni in num_inds_vals}
        het_continuous_runs[ws] = {ni:[] for ni in num_inds_vals}
        hom_contin_exp_runs[ws] = {ni:None for ni in num_inds_vals}

    for num_inds in num_inds_vals:
        for ws in weight_funcs:
            print("running exp hom score: " + ws)
            hom_contin_exp_runs[ws][num_inds] = get_exp_heterogeneity(num_inds,
                                                                      independent_snps,
                                                                      h, weight_funcs[ws])

        for i in range(0, runs):
            print(str(num_inds) + "-" + str(i))
            independent_pop = generate_population(independent_snps, num_inds=num_inds, h=h)
            hetero_pop = generate_population(hetero_snps, num_inds=num_inds, h=h)

            # mean-center phenos
            independent_pop = (independent_pop[0], np.array(independent_pop[1]) - np.mean(independent_pop[1]))
            hetero_pop = (hetero_pop[0], np.array(hetero_pop[1]) - np.mean(hetero_pop[1]))

            het_mean = np.mean(hetero_pop[1])
            het_std = np.std(hetero_pop[1])
            hetero_pop[1][int(num_inds/2):] = np.random.normal(loc=het_mean,
                                                               scale=het_std,
                                                               size=int(num_inds/2))

            for ws in weight_funcs:
                print("num inds:", num_inds, "run:", i, "weight func:", ws)
                hom_continuous_runs[ws][num_inds].append(
                    run_cont_heterogeneity_on_pop(independent_pop, independent_snps,
                                            weight_func=weight_funcs[ws]))
                het_continuous_runs[ws][num_inds].append(
                    run_cont_heterogeneity_on_pop(hetero_pop, hetero_snps,
                                            weight_func=weight_funcs[ws]))

            for bz in bzs:
                print("num inds:", num_inds, "run:", i, "binary thresh:", bz)
                hom_thresh_runs[bz][num_inds].append(
                    run_heterogeneity_on_pop(independent_pop, independent_snps, z=bz))
                het_thresh_runs[bz][num_inds].append(
                    run_heterogeneity_on_pop(hetero_pop, hetero_snps, z=bz))

    with open(file_path, "wb") as f:
        pickle.dump({"hom_thresh_runs": hom_thresh_runs,
                     "het_thresh_runs": het_thresh_runs,
                     "hom_continuous_runs": hom_continuous_runs,
                     "het_continuous_runs": het_continuous_runs,
                     "hom_contin_exp_runs": hom_contin_exp_runs}, f)

def simulate_mult_h(file_path, runs=1, num_snps=100, h_vals=(0.05,0.1,0.15,0.2,0.25,0.3), num_inds=100000, bzs=[-0.6, -0.3, -0.1, 0, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5]):
    weight_funcs = {"log1":log1, "log1p5":log1p5, "log3":log3, "sigmoid":sigmoid, "linear":linear, "step":step, "polynom2":polynom2, "polynom4":polynom4, "polynom6":polynom6}
    hom_thresh_runs = {}
    het_thresh_runs = {}
    hom_continuous_runs = {}
    het_continuous_runs = {}
    hom_contin_exp_runs = {}

    for bz in bzs:
        hom_thresh_runs[bz] = {ni:[] for ni in h_vals}
        het_thresh_runs[bz] = {ni:[] for ni in h_vals}
    for ws in weight_funcs:
        hom_continuous_runs[ws] = {ni:[] for ni in h_vals}
        het_continuous_runs[ws] = {ni:[] for ni in h_vals}
        hom_contin_exp_runs[ws] = {ni:None for ni in h_vals}

    for h in h_vals:
        independent_snps = generate_pss_model_simple(num_snps, h)
        hetero_snps = generate_pss_model_simple(num_snps, h)

        for ws in weight_funcs:
            print("running exp hom score: " + ws)
            hom_contin_exp_runs[ws][h] = get_exp_heterogeneity(num_inds,
                                                               independent_snps,
                                                               h, weight_funcs[ws])

        for i in range(0, runs):
            print(str(num_inds) + "-" + str(i))
            independent_pop = generate_population(independent_snps, num_inds=num_inds, h=h)
            hetero_pop = generate_population(hetero_snps, num_inds=num_inds, h=h)

            # mean-center phenos
            independent_pop = (independent_pop[0], np.array(independent_pop[1]) - np.mean(independent_pop[1]))
            hetero_pop = (hetero_pop[0], np.array(hetero_pop[1]) - np.mean(hetero_pop[1]))

            het_mean = np.mean(hetero_pop[1])
            het_std = np.std(hetero_pop[1])
            hetero_pop[1][int(num_inds/2):] = np.random.normal(loc=het_mean,
                                                               scale=het_std,
                                                               size=int(num_inds/2))


            for ws in weight_funcs:
                print("num inds:", num_inds, "run:", i, "weight func:", ws)
                hom_continuous_runs[ws][h].append(
                    run_cont_heterogeneity_on_pop(independent_pop, independent_snps,
                                            weight_func=weight_funcs[ws]))
                het_continuous_runs[ws][h].append(
                    run_cont_heterogeneity_on_pop(hetero_pop, hetero_snps,
                                            weight_func=weight_funcs[ws]))

            for bz in bzs:
                print("h val:", h, "run:", i, "binary thresh:", bz)
                hom_thresh_runs[bz][h].append(
                    run_heterogeneity_on_pop(independent_pop, independent_snps, z=bz))
                het_thresh_runs[bz][h].append(
                    run_heterogeneity_on_pop(hetero_pop, hetero_snps, z=bz))

    with open(file_path, "wb") as f:
        pickle.dump({"hom_thresh_runs": hom_thresh_runs,
                     "het_thresh_runs": het_thresh_runs,
                     "hom_continuous_runs": hom_continuous_runs,
                     "het_continuous_runs": het_continuous_runs,
                     "hom_contin_exp_runs": hom_contin_exp_runs}, f)


def plot_info_final(file_path, xlabel="heritability"):
    """
    Plot output
    xlabel is adjustible to accommodate both sample size and variance explained x-axes
    """

    disable_binary_threshes = True
    disable_cont_weights = False

    with open(file_path, "rb") as f:
        points = pickle.load(f)
        hom_thresh_runs = points["hom_thresh_runs"]
        het_thresh_runs = points["het_thresh_runs"]
        hom_continuous_runs = points["hom_continuous_runs"]
        het_continuous_runs = points["het_continuous_runs"]
        hom_contin_exp_runs = points["hom_contin_exp_runs"]

    zvals = np.sort(list(hom_thresh_runs.keys()))
    colors = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min(zvals),
                                   vmax=max(zvals)), cmap='cool')
    colors.set_array(zvals)

    weight_order = ['step', 'sigmoid', 'linear', 'log1', 'log1p5', 'log3', 'polynom2', 'polynom4', 'polynom6']

    ws_colors = {'step':'k', 'sigmoid':'orange', 'log1':'r', 'log1p5':'r',
                 'log3':'r', 'linear':'g',
                 'polynom2':'cyan', 'polynom4':'deepskyblue', 'polynom6':'cornflowerblue'}

    ws_styles = {'step':'-', 'sigmoid':'-', 'log1':'--', 'log1p5':'-.',
                 'log3':':', 'linear':'-',
                 'polynom2':'-', 'polynom4':'--', 'polynom6':'-.'}

    fig,ax = plt.subplots(1,1,figsize=(1.6,7))
    plt.gca().set_visible(False)
    print(zvals)
    print(colors)
    cbar = fig.colorbar(colors,ticks=zvals,cax=plt.axes([0.1, 0.1, 0.4, 0.8]))
    cbar.ax.set_yticklabels([str(x) for x in zvals])
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(25)
    if xlabel == "SNP Variance Explained":
        plt.savefig("score_thresh_legend_heritabilities.eps", format="eps", dpi=100)
    elif xlabel == "number of individuals":
        plt.savefig("score_thresh_legend_numinds.eps", format="eps", dpi=100)

    # line plot legend in separate figure
    figlegend, ax = plt.subplots(1,1,figsize=(1.5,2.5))
    plt.gca().set_visible(False)
    ws_labels = sorted(ws_colors.keys())
    legend_lines = [Line2D([0],[0], color=ws_colors[ws],
                           linestyle=ws_styles[ws]) for ws in ws_labels]
    fig_legend = figlegend.legend(legend_lines, ws_labels, frameon=False)
    figlegend.savefig('legendtest.eps')


    show_legend_side = False

    # -----
    # Homogeneous
    # -----
    fig, ax = plt.subplots()
    print("Homogeneous Runs:")
    if not disable_binary_threshes:
        for bz in hom_thresh_runs:
            num_inds_vals = np.sort(list(hom_thresh_runs[bz].keys()))
            score_means = []
            score_stds = []
            for num_inds in num_inds_vals:
                score_means.append(np.mean(hom_thresh_runs[bz][num_inds]))
                score_stds.append(np.std(hom_thresh_runs[bz][num_inds]))
            plt.errorbar(num_inds_vals, score_means, yerr=score_stds, c=colors.to_rgba(bz), capsize=5)

    if not disable_cont_weights:
        for ws in weight_order:
            num_inds_vals = np.sort(list(hom_continuous_runs[ws].keys()))
            score_means = []
            score_stds = []
            for num_inds in num_inds_vals:
                score_means.append(np.mean(hom_continuous_runs[ws][num_inds]))
                score_stds.append(np.std(hom_continuous_runs[ws][num_inds]))
            plt.errorbar(num_inds_vals, score_means, yerr=score_stds, c=ws_colors[ws], linestyle=ws_styles[ws], label=ws, capsize=5)

    if not disable_cont_weights:
        pprint(hom_contin_exp_runs)

        for ws in weight_order:
            num_inds_vals = np.sort(list(hom_contin_exp_runs[ws].keys()))
            scores = [hom_contin_exp_runs[ws][x] for x in num_inds_vals]
            print(num_inds_vals)
            print(scores)
            plt.errorbar(num_inds_vals, scores, c='k', linestyle=':', alpha=1.0, capsize=5)

    if show_legend_side:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.75, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))

    ax.set_ylabel("CLiP-Y score")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if xlabel == "SNP Variance Explained":
        ax.set_xlabel("Variance explained by modeled SNPs")
        plt.savefig("homogeneous_score_heritabilities.eps", format="eps", dpi=1000)
    elif xlabel == "number of individuals":
        ax.set_xlabel("Number of simulated individuals")
        plt.savefig("homogeneous_score_numinds.eps", format="eps", dpi=1000)

    # -----
    # Heterogeneous Means
    # -----
    fig, ax = plt.subplots()
    print("Heterogeneous Runs:")
    if not disable_binary_threshes:
        for bz in het_thresh_runs:
            num_inds_vals = np.sort(list(het_thresh_runs[bz].keys()))
            score_means = []
            score_stds = []
            for num_inds in num_inds_vals:
                score_means.append(np.mean(het_thresh_runs[bz][num_inds]))
                score_stds.append(np.std(het_thresh_runs[bz][num_inds]))
            plt.errorbar(num_inds_vals, score_means, yerr=score_stds, c=colors.to_rgba(bz), capsize=5)

    if not disable_cont_weights:
        for ws in weight_order:
            num_inds_vals = np.sort(list(het_continuous_runs[ws].keys()))
            score_means = []
            score_stds = []
            for num_inds in num_inds_vals:
                score_means.append(np.mean(het_continuous_runs[ws][num_inds]))
                score_stds.append(np.std(het_continuous_runs[ws][num_inds]))
            plt.errorbar(num_inds_vals, score_means, yerr=score_stds, c=ws_colors[ws], linestyle=ws_styles[ws], label=ws, capsize=5)

    ax.set_ylabel("CLiP-Y score")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if xlabel == "SNP Variance Explained":
        ax.set_xlabel("Variance explained by modeled SNPs")
        plt.savefig("heterogeneous_score_heritabilities.eps", format="eps", dpi=1000)
    elif xlabel == "number of individuals":
        ax.set_xlabel("Number of simulated individuals")
        plt.savefig("heterogeneous_score_numinds.eps", format="eps", dpi=1000)

    # -----
    # Heterogeneous score - Homogeneous score
    # -----
    fig, ax = plt.subplots()
    print("Heterogeneous-Homogeneous Runs:")
    if not disable_binary_threshes:
        for bz in het_thresh_runs:
            num_inds_vals = np.sort(list(het_thresh_runs[bz].keys()))
            score_means = []
            score_stds = []
            for num_inds in num_inds_vals:
                score_means.append(np.mean(het_thresh_runs[bz][num_inds]) - \
                                   np.mean(hom_thresh_runs[bz][num_inds]))
                score_stds.append(np.sqrt(np.var(het_thresh_runs[bz][num_inds]) + \
                                          np.var(hom_thresh_runs[bz][num_inds])))
            plt.errorbar(num_inds_vals, score_means, yerr=score_stds, c=colors.to_rgba(bz), capsize=5)

    if not disable_cont_weights:
        for ws in weight_order:
            num_inds_vals = np.sort(list(het_continuous_runs[ws].keys()))
            score_means = []
            score_stds = []
            for num_inds in num_inds_vals:
                score_means.append(np.mean(het_continuous_runs[ws][num_inds]) - \
                                   np.mean(hom_continuous_runs[ws][num_inds]))
                score_stds.append(np.sqrt(np.var(het_continuous_runs[ws][num_inds]) + \
                                          np.var(hom_continuous_runs[ws][num_inds])))
            plt.errorbar(num_inds_vals, score_means, yerr=score_stds, c=ws_colors[ws], linestyle=ws_styles[ws], label=ws, capsize=5)

    ax.set_ylabel("CLiP-Y score")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if xlabel == "SNP Variance Explained":
        ax.set_xlabel("Variance explained by modeled SNPs")
        plt.savefig("bias_adj_score_heritabilities.eps", format="eps", dpi=1000)
    elif xlabel == "number of individuals":
        ax.set_xlabel("Number of simulated individuals")
        plt.savefig("bias_adj_score_numinds.eps", format="eps", dpi=1000)


    # -----
    # vs Threshold
    # -----
    print("Threshold only runs")

    # plot scores as a function of threshold
    fig, (ax1,ax2) = plt.subplots(ncols=2, sharey=True)
    score_means = []
    score_stds = []
    het_means_only = []
    het_stds_only = []
    threshes = np.sort(list(het_thresh_runs.keys()))
    for bz in threshes:
        num_inds = np.max(list(het_thresh_runs[bz].keys()))
        score_means.append(np.mean(het_thresh_runs[bz][num_inds]) -\
                           np.mean(hom_thresh_runs[bz][num_inds]))
        score_stds.append(np.sqrt(np.var(het_thresh_runs[bz][num_inds]) +\
                                  np.var(hom_thresh_runs[bz][num_inds])))
        het_means_only.append(np.mean(het_thresh_runs[bz][num_inds]))
        het_stds_only.append(np.std(het_thresh_runs[bz][num_inds]))
    ax1.errorbar(threshes, score_means, yerr=score_stds, color='k', label='step thresh', capsize=5)

    idx = 0
    for ws in weight_order:
        wscol = ws_colors[ws]
        num_inds = np.max(list(het_thresh_runs[bz].keys()))
        score_mean = np.mean(het_continuous_runs[ws][num_inds]) - \
                     np.mean(hom_continuous_runs[ws][num_inds])
        score_stds = np.sqrt(np.var(het_continuous_runs[ws][num_inds] + \
                             np.var(hom_continuous_runs[ws][num_inds])))

        ax2.errorbar(idx, score_mean, yerr=score_stds, label=ws, color=wscol, marker='o')
        idx += 1
    ax1.set_ylabel("CLiP-Y score")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xlabel("Threshold location (std devs)")
    ax1.set_title("Case/Control Thresholds")
    ax2.get_yaxis().set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xticks(range(len(ws_colors)))
    ax2.set_xticklabels(weight_order, rotation=70)
    ax2.set_title("Continuous Weight Functions")
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.subplots_adjust(wspace=0.05)
    plt.savefig("score_vs_thresh.eps", format="eps", dpi=1000)



    # -----
    # Condensed plot
    # -----
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True, figsize=(11,4.8))
    print("Heterogeneous-Homogeneous Runs:")
    if not disable_binary_threshes:
        for bz in het_thresh_runs:
            num_inds_vals = np.sort(list(het_thresh_runs[bz].keys()))
            score_means = []
            score_stds = []
            for num_inds in num_inds_vals:
                score_means.append(np.mean(het_thresh_runs[bz][num_inds]) - \
                                   np.mean(hom_thresh_runs[bz][num_inds]))
                score_stds.append(np.sqrt(np.var(het_thresh_runs[bz][num_inds]) + \
                                          np.var(hom_thresh_runs[bz][num_inds])))
            ax1.errorbar(num_inds_vals, score_means, yerr=score_stds, c=colors.to_rgba(bz), capsize=5)

    if not disable_cont_weights:
        for ws in weight_order:
            num_inds_vals = np.sort(list(het_continuous_runs[ws].keys()))
            score_means = []
            score_stds = []
            for num_inds in num_inds_vals:
                score_means.append(np.mean(het_continuous_runs[ws][num_inds]) - \
                                   np.mean(hom_continuous_runs[ws][num_inds]))
                score_stds.append(np.sqrt(np.var(het_continuous_runs[ws][num_inds]) + \
                                          np.var(hom_continuous_runs[ws][num_inds])))
            ax1.errorbar(num_inds_vals, score_means, yerr=score_stds, c=ws_colors[ws], linestyle=ws_styles[ws], label=ws, capsize=5)

    ax1.set_ylabel("CLiP-Y score")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_title("Weight Functions")

    print("Threshold only runs")
    # plot scores as a function of threshold
    score_means = []
    score_stds = []
    threshes = np.sort(list(het_thresh_runs.keys()))
    for bz in threshes:
        num_inds = np.max(list(het_thresh_runs[bz].keys()))
        score_means.append(np.mean(het_thresh_runs[bz][num_inds]) -\
                           np.mean(hom_thresh_runs[bz][num_inds]))
        score_stds.append(np.sqrt(np.var(het_thresh_runs[bz][num_inds]) +\
                                  np.var(hom_thresh_runs[bz][num_inds])))
    ax2.errorbar([norm.cdf(x) for x in threshes], score_means, yerr=score_stds, color='k', label='step thresh', capsize=5)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xlabel("PRS Percentile Threshold")
    ax2.set_title("Step Function Thresholds")
    fig.tight_layout()

    if xlabel == "SNP Variance Explained":
        ax1.set_xlabel("Variance explained by modeled SNPs")
        plt.savefig("condensed_performance_wthreshes_heritabilities.eps", format="eps", dpi=1000)
    elif xlabel == "number of individuals":
        ax1.set_xlabel("Number of simulated individuals")
        plt.savefig("condensed_performance_wthreshes_numinds.eps", format="eps", dpi=1000)

    plt.show()

FILE_PATH_H = "data_heritabilities.pickle"
FILE_PATH_N = "data_num_inds.pickle"

if __name__=="__main__":
    if not os.path.exists(FILE_PATH_H):
        simulate_mult_h(FILE_PATH_H, runs=20, num_snps=10, h_vals=(0.01,0.05,0.1,0.15,0.2,0.25,0.3), num_inds=100000, bzs=[-1.6,-1.4,-1.2,-1.0,-0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0, 2.2])


    if not os.path.exists(FILE_PATH_N):
        simulate_mult_n(FILE_PATH_N, runs=20, num_snps=10, h=0.1, num_inds_vals=(5000, 10000, 20000, 50000, 100000), bzs=[-1.6,-1.4,-1.2,-1.0,-0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0, 2.2])

    plot_info_final(FILE_PATH_H, xlabel="SNP Variance Explained")
    plot_info_final(FILE_PATH_N, xlabel="number of individuals")

