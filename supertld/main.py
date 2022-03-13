# _*_ coding: utf-8 _*_
"""
Time:     2021/12/13 21:50
Author:   ZHANG Yuwei
Version:  V 0.1
File:     main.py
Describe:
"""

import numpy as np
import pandas as pd
import warnings, subprocess
warnings.filterwarnings("ignore")
from .smurf import SMURF, SMURF_SYM
from .KR_norm_juicer import KRnorm_asym, KRnorm_sym
from .AsymToSym import asym_to_sym
from .utils import heatmap


class SupertadSparse():
    def __init__(self, chrom="chr1", resolution=100000, norm=True, smurf=True,
                 supertad="/home/zhangyuwei/hic_TAD/SuperTAD_bakcup/build/SuperTAD", dataweight=0., verbose=False):
        self.chrom = chrom
        self.resolution = resolution
        self.norm = norm
        self.smurf = smurf
        self.supertadPath = supertad
        self.dataweight = dataweight
        self.verbose = verbose

    def pipeline(self, inputMatrix, outpath="./norm_matrix.txt", noise_model="Fano", learningRate=1e-6, lambda_regulation=None, hic=None, alpha=None):
        if inputMatrix.shape[0] == inputMatrix.shape[1] and np.allclose(inputMatrix, inputMatrix.T):    # check if symmetric
            # symmetric
            if self.verbose:
                heatmap(inputMatrix, "./sym_00_raw")
                heatmap(inputMatrix, "./sym_00_raw_diag", diag=True)

            if self.norm:
                print("Start to perform KR norm on the raw data.")
                imputed_matrix, _ = KRnorm_sym(inputMatrix, oneNorm=False, verbose=self.verbose)
                if self.verbose:
                    np.savetxt("./sym_01_kr.txt", imputed_matrix)
                    heatmap(imputed_matrix, "./sym_01_kr")
                    heatmap(imputed_matrix, "./sym_01_kr_diag", diag=True)
            else:
                imputed_matrix = inputMatrix

            if self.smurf:
                if lambda_regulation:
                    print("Start to impute the symmetric contact map with SMURF with model {} and lambda {}.".format(noise_model, lambda_regulation))
                else:
                    lambda_regulation = 0.01
                    print("Start to impute the symmetric contact map with SMURF with model {} and lambda {}.".format(
                        noise_model, lambda_regulation))
                operator = SMURF_SYM(n_features=None, lambda2=lambda_regulation, estimate_only=True, noise_model=noise_model,
                                           weight=self.dataweight, verbose=self.verbose, alpha=learningRate)
                imputed_matrix = operator.smurf_impute(pd.DataFrame(imputed_matrix))

                imputed_matrix = imputed_matrix.values
                if self.verbose:
                    np.savetxt("./sym_02_smurf.txt", imputed_matrix)
                    heatmap(imputed_matrix, "./sym_02_smurf")
                    heatmap(imputed_matrix, "./sym_02_smurf_diag", diag=True)
            else:
                imputed_matrix = imputed_matrix

            if hic != None: # data integration
                hicMatrix = np.loadtxt(hic)
                if hicMatrix.shape != imputed_matrix.shape:
                    raise ValueError("The two contact maps' shape doesn't match.")
                if alpha:
                    matrix = alpha * imputed_matrix + (1 - alpha) * hicMatrix
                    np.savetxt(outpath + str(alpha), matrix)
                    return outpath+str(alpha), self.Perform_SuperTAD(outpath + str(alpha))
                else:
                    outputList = []
                    tldList = []
                    for i in np.arange(0, 1.05, 0.05):
                        alpha = i
                        matrix = alpha * imputed_matrix + (1 - alpha) * hicMatrix
                        np.savetxt(outpath+str(i), matrix)
                        tldList.append(self.Perform_SuperTAD(outpath+str(i)))
                        outputList.append(outpath+str(i))
                    return outputList, tldList
            else:
                np.savetxt(outpath, imputed_matrix)
                return outpath, self.Perform_SuperTAD(outpath)

        else:   # asymmetric
            if self.smurf:
                if lambda_regulation:
                    print("Start to impute the asymmetric contact map with SMURF with model {} and lambda {}.".format(noise_model, lambda_regulation))
                else:
                    lambda_regulation = 1e-5
                    print("Start to impute the asymmetric contact map with SMURF with model {} and lambda {}.".format(
                        noise_model, lambda_regulation))
                operator = SMURF(n_features=None, normalize=self.norm, lambda2=lambda_regulation, estimate_only=True,
                                       noise_model=noise_model, weight=self.dataweight, verbose=self.verbose, alpha=learningRate)
                imputed_matrix, G, H = operator.smurf_impute(pd.DataFrame(inputMatrix))
                imputed_matrix = imputed_matrix.values
                if self.verbose:
                    np.savetxt("./asym_smurf.txt", imputed_matrix)
                    heatmap(imputed_matrix, "./asym_smurf")
                    heatmap(imputed_matrix, "./asym_smurf_diag", diag=True)
            else:
                imputed_matrix = inputMatrix

            if hic != None: # data integration
                if alpha:
                    matrix = asym_to_sym(imputed_matrix, extraNorm=True, hic=hic, alpha=alpha)
                    np.savetxt(outpath+str(alpha), matrix)
                    return outpath+str(alpha), self.Perform_SuperTAD(outpath+str(alpha))
                else:
                    outputList = []
                    tldList = []
                    for i in np.arange(0, 1.05, 0.05):
                        matrix = asym_to_sym(imputed_matrix, extraNorm=True, hic=hic, alpha=i)
                        np.savetxt(outpath+str(i), matrix)
                        tldList.append(self.Perform_SuperTAD(outpath+str(i)))
                        outputList.append(outpath+str(i))
                    return outputList, tldList
            else:
                matrix = asym_to_sym(imputed_matrix, extraNorm=True)
                np.savetxt(outpath, matrix)
                result = self.Perform_SuperTAD(outpath)
                return outpath, result

    def Perform_SuperTAD(self, inputPath, hu=1, hd=1, sparse=True):  #sparse mode
        if sparse:
            job = [self.supertadPath, "multi_2d", inputPath, "--hu", str(hu), "--hd", str(hd), "--chrom1", self.chrom, "-r", str(self.resolution), "-s"]
        else:
            job = [self.supertadPath, "multi_2d", inputPath, "--hu", str(hu), "--hd", str(hd), "--chrom1", self.chrom, "-r", str(self.resolution)]
        p = subprocess.Popen(job, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        status = p.wait()
        if sparse:
            return inputPath + ".multi2D_AllH2_sparse.tsv"
        else:
            return inputPath + ".multi2D_AllH2.tsv"

