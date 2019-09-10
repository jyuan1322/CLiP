import numpy as np
from numpy.linalg import inv
from scipy.stats import norm, chi2
import sys

class Heterogeneity:
    def het(self, num_snps):
        self.latest = {}
        self.index = 0
        self.numer = 0.0
        self.denom = 0.0
        self.num_snps = num_snps
        self.z = np.zeros(self.num_snps)

    def get_full_mat(self):
        return self.R

    def get_values(self, clist):
        numer = 0.0
        denom = 0.0
        num_snps = len(clist)
        for i in range(num_snps):
            for j in range(i+1,num_snps):
                wij = self.z[clist[i]] * self.z[clist[j]]
                yij = self.Y[clist[i],clist[j]]
                numer += wij * yij
                denom += wij * wij
        score = numer / np.sqrt(denom)
        return score

    def get_weights(self):
        return self.z

class Heterogeneity_GWAS(Heterogeneity):
    def het(self, cases, controls):
        super().het(cases.shape[1])
        N = float(len(cases))
        Np = float(len(controls))
        self.R = np.corrcoef(cases.T)
        self.Rp = np.corrcoef(controls.T)
        self.Y = np.sqrt(N*Np/(N+Np)) * (self.R-self.Rp)
        for i in range(self.num_snps):
            p_case = np.sum(cases[:,i]) / (2*cases.shape[0])
            p_control = np.sum(controls[:,i]) / (2*controls.shape[0])
            gamma = p_case/(1-p_case) / (p_control/(1-p_control))
            self.z[i] = np.sqrt(p_control*(1-p_control)) * (gamma-1) / (p_control*(gamma-1) + 1)

class Heterogeneity_TWAS(Heterogeneity):
    def het(self, cases, controls):
        super().het(cases.shape[1])
        N = float(len(cases))
        Np = float(len(controls))
        self.R = np.corrcoef(cases.T)
        self.Rp = np.corrcoef(controls.T)
        self.Y = np.sqrt(N*Np/(N+Np)) * (self.R-self.Rp)
        n = controls.shape[0]
        for i in range(self.num_snps):
            mui_control = np.mean(controls[:,i])
            mui_case = np.mean(cases[:,i])
            stdi_control = np.std(controls[:,i])
            self.z[i] = (mui_case - mui_control)/stdi_control

    def het_expcorr(self, N, Np, expcorr, contcorr, mu_case, mu_cont, sd_cont):
        super().het(expcorr.shape[0])
        self.R = expcorr
        self.Rp = contcorr
        self.Y = np.sqrt(N*Np/(N+Np)) * (self.R-self.Rp)
        for i in range(self.num_snps):
            self.z[i] = (mu_case[i] - mu_cont[i])/sd_cont[i]

    def het_expcorr2(self, N, Np, expcorr, controls, mu_case):
        super().het(expcorr.shape[0])
        self.R = expcorr
        self.Rp = np.corrcoef(controls, rowvar=False)
        self.Y = np.sqrt(N*Np/(N+Np)) * (self.R-self.Rp)
        for i in range(self.num_snps):
            mu_cont = np.mean(controls[:,i])
            sd_cont = np.std(controls[:,i])
            self.z[i] = (mu_case[i] - mu_cont)/sd_cont
