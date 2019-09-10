import numpy as np
import matopr

# Add noise to the data to conform to variance explained
def add_heritability(std, h, data):
    dev = np.sqrt(((1 - h) / h) * std**2)
    noise = np.random.normal(0, dev, len(data))
    return data + noise

# generate the SNP vectors of individuals according to the specified RAF
def simulate_individual(num, freqs):
	num = int(num)
	inds = np.empty([num, len(freqs)])
	for i, p in enumerate(freqs):
		sprobs = [(1-p)*(1-p), 2*p*(1-p), p*p]
		inds[:, i] = np.random.choice(3,size=num,p=sprobs)
	return inds

# Get the relevant parameters for simulation given the weights and the set up
# This partitions the alpha weight accoding to different sets of genotype.
def get_params(freqs, gen2expr_wgtmat, alpha, perm, gene_props):
    # expression weights for each disease (sub-phenotype)
    # the elements in each row of subalpha that does not correspond to the
    # disease of that row will be set to 0
    subalpha = np.zeros([len(gene_props), len(alpha)])
    start = 0
    for i in range(len(gene_props)):
        end = int(start + gene_props[i] * len(alpha))
        subalpha[i, perm[start:end]] = alpha[perm[start:end]]
        start = end
    return subalpha

def get_fixed_expression_params(num_snps, num_expr, g2wherit, heritability):
    # set g2wherit to a fixed value across variables if a particular var explained is desired
    # freqs are set to a constant 0.5 if fixed
    if g2wherit is not None:
        val = np.sqrt(g2wherit / num_snps * 2)
        gen2expr_wgtmat = np.ones(shape=(num_expr, num_snps)) * val
        freqs = np.random.uniform(low=0.5, high=0.5, size=num_snps)
    else:
        gen2expr_wgtmat = np.random.normal(loc=0.0,scale=0.07, size=(num_expr, num_snps))
        freqs = np.random.uniform(low=0.05, high=0.95, size=num_snps)

    if heritability is not None:
        geno_vars = 2.0 * np.multiply(freqs, 1-freqs)
        gcontrib = 0
        for i in range(num_expr):
            for j in range(i+1, num_expr):
                gcontrib += 2 * np.sum(np.multiply(
                                  np.multiply(gen2expr_wgtmat[i,:],
                                              gen2expr_wgtmat[j,:]),geno_vars))
        val = np.sqrt(heritability / (num_expr + gcontrib))
        alpha = np.array([val] * num_expr)
    else:
        alpha = np.random.normal(loc=0.0, scale=0.1, size=num_expr)

    # check heritabilities
    g2wherit = []
    for i in range(num_expr):
        g2wherit.append(np.sum(np.multiply(np.square(gen2expr_wgtmat[i,:]), 2*np.multiply(freqs, 1.0-freqs))))
    geno_vars = 2.0 * np.multiply(freqs, 1-freqs)
    heritability = np.sum(np.multiply(np.square(alpha), np.ones(shape=num_expr)))
    for i in range(num_expr):
        for j in range(i+1, num_expr):
            heritability += 2 * np.sum(np.multiply(
                              np.multiply(gen2expr_wgtmat[i,:],
                                          gen2expr_wgtmat[j,:]),geno_vars)) * alpha[i] * alpha[j]
    return freqs, gen2expr_wgtmat, alpha, g2wherit, heritability



def generate_cohort(num_snps, num_expr, gen2expr_wgtmat, freqs, alpha, gene_props, case_sizes, heritability, g2wherit, thresh, use_logistic=False):
    """
    case_sizes should be a list, i.e. [5000,5000]
    """

    # Permute the values of alpha
    # If simulating, this step is unnecessary
    # perm = np.random.permutation(len(alpha))
    perm = range(len(alpha))

    subalpha = get_params(freqs, gen2expr_wgtmat, alpha, perm, gene_props)
    case_genos, cases = get_cases_unnorm(case_sizes,
                                           freqs,
                                           gen2expr_wgtmat,
                                           subalpha,
                                           heritability,
                                           g2wherit,
                                           thresh)
    return case_genos, cases

"""
unnormalized: all expression must add error to sum to standard normal distribution
"""
def get_cases_unnorm(case_sizes, freqs, gen2expr_wgtmat, subalpha, heritability, g2wherit, thresh=1.8, batch=3000, max_elements=int(2**20)):
    cases = []
    for i in range(len(case_sizes)):
        cases.append(np.empty([case_sizes[i], gen2expr_wgtmat.shape[0]]))

    # also store genotypes for generated indivs, if effect sizes need to be estimated
    case_genos = []
    for i in range(len(case_sizes)):
        case_genos.append(np.empty([case_sizes[i], gen2expr_wgtmat.shape[1]]))


    all_count = 0 # cases for all subphenotypes
    cur_count = 0
    cur_case_size = [0] * len(case_sizes)
    for case_size in case_sizes:
        all_count += case_size

    while cur_count < all_count:
        individuals = simulate_individual(batch, freqs)
        expression = matopr.blockwise_dot(gen2expr_wgtmat, individuals.T, max_elements=max_elements).T
        # add heritability to expression
        num_expr, num_snps = gen2expr_wgtmat.shape
        for i in range(num_expr):
            # mean-center the expression values
            expr_mean = 2 * np.sum(np.multiply(gen2expr_wgtmat[i,:], freqs))
            expression[:,i] -= expr_mean
            # add error (to variance=1 for all variables)
            expression[:,i] += np.random.normal(0, np.sqrt(1.0-g2wherit[i]), expression.shape[0])

        # add error to risk
        risk = expression.dot(subalpha.T)
        risk += np.random.normal(0, np.sqrt(1.0-heritability), risk.shape)

        for i in range(risk.shape[0]):
            for j in range(risk.shape[1]):
                # liability threshold
                if risk[i,j] > thresh and cur_case_size[j] < case_sizes[j]:
                    cases[j][cur_case_size[j], :] = expression[i]

                    case_genos[j][cur_case_size[j], :] = individuals[i,:]
                    cur_count += 1
                    cur_case_size[j] += 1

        print("[CASE] generated", cur_count, "out of", all_count)
    return case_genos, cases

"""    
unnormalized: all expression must add error to sum to standard normal distribution
"""
def simulate_g2e_unnorm(num_indivs, freqs, gen2expr_wgtmat, g2wherit, max_elements=int(2**20)):
    individuals = simulate_individual(num_indivs, freqs)
    expression = matopr.blockwise_dot(gen2expr_wgtmat, individuals.T, max_elements=max_elements).T
    
    num_expr, num_snps = gen2expr_wgtmat.shape

    for i in range(num_expr):
        # mean-center the expression values
        expr_mean = 2 * np.sum(np.multiply(gen2expr_wgtmat[i,:], freqs))
        expression[:,i] -= expr_mean
        # add error (to variance=1 for all variables)
        expression[:,i] += np.random.normal(0, np.sqrt(1.0-g2wherit[i]), num_indivs)
    return individuals, expression

