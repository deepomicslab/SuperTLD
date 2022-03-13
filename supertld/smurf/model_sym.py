# _*_ coding: utf-8 _*_
"""
Time:     2021/12/2 21:47
Author:   ZHANG Yuwei
Version:  V 0.2
File:     model_sym.py
Describe: SMURF for symmetric matrix
"""

from .lossFunc_sym import *
from .initialParams import *
import pandas as pd
from .optimize import CVOptimize_sym, FanoOptimize_sym, CCVOptimize_sym
from . import utiles
from .cell_circle import CellCircle
import warnings

warnings.filterwarnings("ignore")


class SMURF():
    def __init__(self, n_features=None, steps=1000, alpha=1e-5, eps=1e-3, lambda2=0.1, noise_model="Fano",
                 calculateIntialNoiseFactor=False, estimate_only=False, weight=0., verbose=False):

        self.K = n_features
        if self.K:
            self.batchSize = n_features * 10
        else:
            self.batchSize = None
        self.steps = steps
        self.alpha = alpha
        self.eps = eps
        self.lambda2 = lambda2
        self.noise_model = noise_model
        self.calculateLossFunc = True
        self.estmate_only = estimate_only
        self.calculateIntialNoiseFactor = calculateIntialNoiseFactor
        self.dataweightControl = weight  # add a controller on the data weighted gradient, y ** weight
        self.verbose = verbose

    def _check_params(self):
        """Check parameters

        This allows us to fail early - otherwise certain unacceptable
        parameter choices, such as n='10.5', would only fail after
        minutes of runtime.

        Raises
        ------
        ValueError : unacceptable choice of parameters
        """
        utiles.check_positive(n_features=self.K, eps=self.eps, batchsize=self.batchSize, alpha=self.alpha,
                              steps=self.steps)
        utiles.check_int(n_features=self.K, steps=self.steps)
        utiles.check_between(v_min=0, v_max=min(self.genes, self.cells), n_features=self.K)
        utiles.check_bool(iteration=self.calculateIntialNoiseFactor)
        utiles.check_noise_model(noise_model=self.noise_model)

    def MFConstantVariance(self):   # constance variance for each bin, v_b
        G = self.G.copy()
        A = self.A.copy()
        u = np.dot(G, G.T)

        if self.calculateIntialNoiseFactor:
            Vb = getv(self.A, u)    # (Nbin, 1)
            v = np.sqrt(np.dot(Vb, Vb.transpose()))  # v = sqrt(vi*vj)
        else:
            Vb = np.ones((self.bins, 1))
            v = np.sqrt(np.dot(Vb, Vb.transpose()))

        LossFuncGre = 0
        LossFunc = 0
        Lossf = []

        nonZerosInd = np.nonzero(np.triu(A))
        numNonZeros = nonZerosInd[0].size
        nonZeroElem = np.concatenate((nonZerosInd[0].reshape(numNonZeros, 1), nonZerosInd[1].reshape(numNonZeros, 1)),
                                     axis=1)

        for step in range(self.steps):

            if self.calculateLossFunc:
                for element in nonZeroElem:
                    b1 = element[0]
                    b2 = element[1]

                    LossFuncCG = LossFunctionConstantVariance(u[b1][b2], v[b1][b2], A[b1][b2], G, b1, b2, self.lambda2, self.dataweightControl, self.verbose)
                    LossFunc = LossFunc + LossFuncCG
            else:
                LossFunc = (step + 1) * (self.eps + 1)

            if abs(LossFunc - LossFuncGre) < self.eps:
                Lossf.append(LossFunc)
                # print("already converge")
                break
            else:
                Lossf.append(LossFunc)
                LossFuncGre = LossFunc
                LossFunc = 0

                np.random.shuffle(nonZeroElem)

                batchElements = nonZeroElem[0:self.batchSize - 1, :]

                for element in batchElements:
                    b1 = element[0]
                    b2 = element[1]
                    if u[b1][b2] <= 0.01:
                        u[b1][b2] = 0.01
                    if Vb[b1][0] <= 1e-09:
                        Vb[b1][0] = 1e-09
                    if Vb[b2][0] <= 1e-09:
                        Vb[b2][0] = 1e-09
                    v[b1][b2] = np.sqrt(Vb[b1][0]*Vb[b2][0])

                    dG1, dG2, dV1, dV2 = CVOptimize_sym(A, G, u, Vb, v, b1, b2, self.lambda2, self.dataweightControl)
                    G[b1, :] = G[b1, :] + self.alpha * dG1
                    G[b2, :] = G[b2, :] + self.alpha * dG2
                    Vb[b1, :] = Vb[b1, :] + self.alpha * dV1
                    Vb[b2, :] = Vb[b2, :] + self.alpha * dV2

                u = np.dot(G, G.T)
                v = np.sqrt(np.dot(Vb, Vb.transpose()))
            print("number of iteration: ", step + 1, "/", self.steps, LossFuncGre)
        u = np.dot(G, G.T)
        u[u < 0] = 0
        return u, G

    def MFFano(self):
        G = self.G.copy()
        A = self.A.copy()
        u = np.dot(G, G.T)
        alpha = self.alpha

        if self.calculateIntialNoiseFactor:
            bg = getb(self.A, u)    # (Nbin, 1)
        else:
            bg = np.ones((self.bins, 1)) * 1.0

        b = np.sqrt(np.dot(bg, bg.transpose())) # (Nbin, Nbin)

        LossFunc = 0
        LossFuncGre = 0
        Lossf = []

        nonZerosInd = np.nonzero(np.triu(A))
        numNonZeros = nonZerosInd[0].size
        # print(nonZerosInd, numNonZeros)
        nonZeroElem = np.concatenate((nonZerosInd[0].reshape(numNonZeros, 1), nonZerosInd[1].reshape(numNonZeros, 1)),
                                     axis=1)

        for step in range(self.steps):
            #print(G, G.shape)
            #print(bg, bg.shape)
            if self.calculateLossFunc:
                for element in nonZeroElem:
                    b1 = element[0]
                    b2 = element[1]

                    LossFuncCG = LossFunctionFano(u[b1][b2], b[b1][b2], A[b1][b2], G, b1, b2, self.lambda2, self.dataweightControl, self.verbose)
                    LossFunc = LossFunc + LossFuncCG
            else:
                LossFunc = (step + 1) * self.eps

            if abs(LossFunc - LossFuncGre) < self.eps or np.isnan(LossFunc):
                Lossf.append(LossFunc)
                print("already converge")
                break
            else:
                Lossf.append(LossFunc)
                LossFuncGre = LossFunc
                LossFunc = 0

                np.random.shuffle(nonZeroElem)

                batchElements = nonZeroElem[0:self.batchSize - 1, :]
                for element in batchElements:
                    b1 = element[0]
                    b2 = element[1]
                    if u[b1][b2] <= 0.01:
                        u[b1][b2] = 0.01
                    if bg[b1][0] <= 1e-09:
                        bg[b1][0] = 1e-09
                    if bg[b2][0] <= 1e-09:
                        bg[b2][0] = 1e-09
                    b[b1][b2] = np.sqrt(bg[b1][0] * bg[b2][0])

                    dG1, dG2, db1, db2 = FanoOptimize_sym(A, G, u, bg, b, b1, b2, self.lambda2, self.dataweightControl)
                    # print(dG1, dG2, db1, db2, A[b1][b2], self.lambda2)
                    G[b1, :] = G[b1, :] + alpha * dG1
                    G[b2, :] = G[b2, :] + alpha * dG2
                    bg[b1, :] = bg[b1, :] + alpha * db1
                    bg[b2, :] = bg[b2, :] + alpha * db2

                u = np.dot(G, G.T)
                b = np.sqrt(np.dot(bg, bg.transpose()))

                alpha = alpha * (1 - float(step / self.steps))
                print("number of iteration: ", step + 1, "/", self.steps, LossFuncGre)
        u = np.dot(G, G.T)
        u[u < 0] = 0

        if Lossf[-1] > Lossf[0]:
            return u, G
        elif self.dataweightControl < 2:
            self.dataweightControl += 0.5
            print("Fail to perform SMURF with this setting on the data, obj get worse. change weight as {}, learningRate as {}".format(self.dataweightControl, self.alpha))
            return self.MFFano()
        else:
            self.dataweightControl = 0
            self.alpha = 0.1 * self.alpha
            print("Fail to perform SMURF with this setting on the data, obj get worse. change weight as {}, learningRate as {}".format(self.dataweightControl, self.alpha))
            return self.MFFano()

    def MFConstCoeffiVariation(self):
        G = self.G.copy()
        A = self.A.copy()
        u = np.dot(G, G.T)

        if self.calculateIntialNoiseFactor:
            ag = geta(self.A, u)    # (Nbin, 1)
        else:
            ag = np.ones((self.bins, 1))

        a = np.sqrt(np.dot(ag, ag.transpose())) # (Nbin, Nbin)

        LossFunc = 0
        LossFuncGre = 0
        Lossf = []

        nonZerosInd = np.nonzero(np.triu(A))
        numNonZeros = nonZerosInd[0].size
        nonZeroElem = np.concatenate((nonZerosInd[0].reshape(numNonZeros, 1), nonZerosInd[1].reshape(numNonZeros, 1)),
                                     axis=1)

        for step in range(self.steps):
            if self.calculateLossFunc:
                for element in nonZeroElem:
                    b1 = element[0]
                    b2 = element[1]
                    LossFuncCG = LossFunctionConstantCoefficientVariation(u[b1][b2], a[b1][b2], A[b1][b2], G, b1, b2, self.lambda2, self.dataweightControl, self.verbose)
                    LossFunc = LossFunc + LossFuncCG
            else:
                LossFunc = (step + 1) * self.eps

            if abs(LossFunc - LossFuncGre) < self.eps:
                Lossf.append(LossFunc)
                print("already converge")
                break
            else:
                Lossf.append(LossFunc)
                LossFuncGre = LossFunc
                LossFunc = 0

                np.random.shuffle(nonZeroElem)

                batchElements = nonZeroElem[0:self.batchSize - 1, :]
                for element in batchElements:
                    b1 = element[0]
                    b2 = element[1]
                    if u[b1][b2] <= 0.01:
                        u[b1][b2] = 0.01
                    if ag[b1][0] <= 1e-09:
                        ag[b1][0] = 1e-09
                    if ag[b2][0] <= 1e-09:
                        ag[b2][0] = 1e-09
                    a[b1][b2] = np.sqrt(ag[b1][0]*ag[b2][0])

                    dG1, dG2, da1, da2 = CCVOptimize_sym(A, G, u, ag, a, b1, b2, self.lambda2, self.dataweightControl)
                    G[b1, :] = G[b1, :] + self.alpha * dG1
                    G[b2, :] = G[b2, :] + self.alpha * dG2
                    ag[b1, :] = ag[b1, :] + self.alpha * da1
                    ag[b2, :] = ag[b2, :] + self.alpha * da2

                u = np.dot(G, G.T)
                a = np.sqrt(np.dot(ag, ag.transpose()))
            print("number of iteration: ", step + 1, "/", self.steps, LossFuncGre)
        u = np.dot(G, G.T)
        u[u < 0] = 0

        return u, G

    def smurf_impute(self, initialDataFrame):
        self.A = initialDataFrame.values
        self.binNames = initialDataFrame._stat_axis.values.tolist()
        self.bins = self.A.shape[0]
        if not self.K:
            self.K = self.bins
            self.batchSize = self.K * 10
        self.genes = self.cells = self.bins

        print("Running SCEnd_sym on {} bins, with {} dimension".format(self.bins, self.K))

        self.G = initialMatrix_sym(self.A, self.K)  # (#bin, #feature)

        self._check_params()

        print("preprocessing data...")

        if self.noise_model == "CV":  # constance variance for each bin, v_b
            u, G = self.MFConstantVariance()

        elif self.noise_model == "Fano":
            u, G = self.MFFano()

        elif self.noise_model == "CCV":
            u, G = self.MFConstCoeffiVariation()

        newDataFrame = pd.DataFrame(u, index=self.binNames)

        res = {}

        res["estimate"] = newDataFrame
        res["bin latent factor matrix"] = pd.DataFrame(G, index=self.binNames, columns=None)

        self.glfm = G

        if self.estmate_only:
            return res["estimate"]
        else:
            return res

    def smurf_cell_circle(self, cells_data=None, n_neighbors=20, min_dist=0.01, major_axis=3, minor_axis=2, k=0.2):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.k = k

        if cells_data:
            data = cells_data
        else:
            if self.clfm.all():
                data = self.clfm
            else:
                raise AttributeError("Cells Data Expected")

        cell_circle_mapper = CellCircle(n_neighbors=self.n_neighbors, min_dist=self.min_dist)
        res = cell_circle_mapper.cal_cell_circle(data, a=self.major_axis, b=self.minor_axis, k=0.2)

        return res




























