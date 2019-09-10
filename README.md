CLiP is a statistical tool to detect genotypic heterogeneity in single case/control GWAS cohorts. Heterogeneity among cases may arise due to a phenotype comprising multiple latent sub-phenotypes. CLiP calculates correlations between associated SNPs and compares them to expected values from summary statistics.

# Requirements
* Python3
* numpy
* scipy
* matplotlib
* To display dynamic line labels in `CLiP_input.py`, the `labellines` package is required and can be installed by the following command
```
pip install matplotlib-label-lines
```

# Simulations
To reproduce figures in the manuscript, simply execute
```
python3 run.py
```
within any folder pertaining to the figure. This will search the current figure directory for a pickle file with stored simulation results. To re-run the simulation, delete or rename the pickle file. Certain scripts will allow user input of simulation parameters - run with the option `--help` to view input options.

# Data Analysis
By default, `CLiP_input.py` will plot values in the pickle file `cohort_scores.pickle`. To analyze a new case/control data set, delete or rename the pickle file and run the following command
```
python3 CLiP_input.py --snp-path <snp-path> --geno-path <geno-path> --pheno-path <pheno-path> --file-list <file-list>
```
* `<snp-path>` is a comma-delimited file containing summary statistics information on included SNPs. The columns of this file are as follows. The column "A12" lists the reference and alternate alleles without separators, e.g. "TC." Frequencies and ORs pertain to the first listed allele of column "A12."
```
Rank|Index SNP|A12|Frqcase|Frqcontrol|Chr|Position|Combined OR|Combined P|Discovery OR|Discovery P|Replication OR|Replication P
```
* `<geno-path>` is a directory containing genotype data in Oxford .haps format, one file per case/control cohort.
* `<pheno-path>` is a directory containing pheontype labels per individual in .sample format, one file per case/control cohort.
* `<file-list>` is a comma-delimited file pairing file paths in `<geno-path>` and `<pheno-path>'. The format is
```
cohort name|.haps file|.sample file
```
