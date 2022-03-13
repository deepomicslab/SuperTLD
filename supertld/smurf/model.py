# _*_ coding: utf-8 _*_
"""
Time:     2021/7/22 17:29
Author:   WANG Bingchen
Version:  V 0.1
File:     model.py
Describe: 
"""

from .lossFunc import *
from .initialParams import *
import pandas as pd
from .optimize import CVOptimize, FanoOptimize, CCVOptimize
from . import utiles
from .cell_circle import CellCircle
import warnings
warnings.filterwarnings("ignore")


class SMURF():
    def __init__(self, n_features=None, steps=1000, alpha=1e-6, eps=1e-3, lambda2=0.1, noise_model="Fano",
                 normalize=True, calculateIntialNoiseFactor=False, estimate_only=False, weight=0., verbose=False):

        self.K = n_features
        if self.K:
            self.batchSize = n_features * 10
        else:
            self.batchSize = None
        self.steps = steps
        self.alpha = alpha
        self.eps = eps
        self.normalize = normalize
        self.lambda2 = lambda2
        self.noise_model = noise_model
        self.calculateLossFunc = True
        self.estmate_only = estimate_only
        self.calculateIntialNoiseFactor = calculateIntialNoiseFactor
        self.dataweightControl = weight # add a controller on the data weighted gradient, y ** weight
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
        utiles.check_bool(normalize=self.normalize)
        utiles.check_bool(iteration=self.calculateIntialNoiseFactor)
        utiles.check_noise_model(noise_model=self.noise_model)

    def MFConstantVariance(self):
        G = self.G.copy()
        H = self.H.copy()
        A = self.A.copy()
        u = np.dot(G, H)
        Vc = np.ones((1, self.cells)) * 1.0

        if self.calculateIntialNoiseFactor:
            Vg = getv(self.A, u)
            v = np.dot(Vg, Vc)
        else:
            Vg = np.ones((self.genes, 1))
            v = np.dot(Vg, Vc)

        LossFuncGre = 0
        LossFunc = 0
        Lossf = []

        nonZerosInd = np.nonzero(A)
        numNonZeros = nonZerosInd[0].size
        nonZeroElem = np.concatenate((nonZerosInd[0].reshape(numNonZeros, 1), nonZerosInd[1].reshape(numNonZeros, 1)),
                                     axis=1)

        for step in range(self.steps):

            if self.calculateLossFunc:
                for element in nonZeroElem:
                    g = element[0]
                    c = element[1]
                    LossFuncCG = LossFunctionConstantVariance(u[g][c], v[g][c], A[g][c], G, H, g, c, self.lambda2, self.dataweightControl, self.verbose)
                    LossFunc = LossFunc + LossFuncCG    # minimize LossFunc
            else:
                LossFunc = (step+1)*(self.eps+1)


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
                    g = element[0]
                    c = element[1]
                    if u[g][c] <= 0.01:
                        u[g][c] = 0.01
                    if Vg[g][0] <= 1e-09:
                        Vg[g][0] = 1e-09

                    dG, dH, dVg = CVOptimize(A, G, H, u, Vg, Vc, v, g, c, self.lambda2, self.dataweightControl)
                    G[g, :] = G[g, :] + self.alpha * dG
                    H[:, c] = H[:, c] + self.alpha * dH
                    Vg[g, :] = Vg[g, :] + self.alpha * dVg

                u = np.dot(G, H)
                v = np.dot(Vg, Vc)
            print("number of iteration: ", step+1, "/", self.steps, LossFuncGre)
        u = np.dot(G, H)
        u[u < 0] = 0
        return u, G, H

    def MFFano(self):
        G = self.G.copy()
        H = self.H.copy()
        A = self.A.copy()
        u = np.dot(G, H)
        bc = np.ones((1, self.cells)) * 1.0
        alpha = self.alpha

        if self.calculateIntialNoiseFactor:
            bg = getb(self.A, u)
        else:
            bg = np.ones((self.genes, 1)) * 1.0

        b = np.dot(bg, bc)

        LossFunc = 0
        LossFuncGre = 0
        Lossf = []

        nonZerosInd = np.nonzero(A)
        numNonZeros = nonZerosInd[0].size
        nonZeroElem = np.concatenate((nonZerosInd[0].reshape(numNonZeros, 1), nonZerosInd[1].reshape(numNonZeros, 1)),
                                     axis=1)

        for step in range(self.steps):
            if self.calculateLossFunc:
                for element in nonZeroElem:
                    g = element[0]
                    c = element[1]

                    LossFuncCG = LossFunctionFano(u[g][c], b[g][c], A[g][c], G, H, g, c, self.lambda2, self.dataweightControl, self.verbose)
                    LossFunc = LossFunc + LossFuncCG
            else:
                LossFunc = (step + 1)*self.eps


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
                    g = element[0]
                    c = element[1]
                    if u[g][c] <= 0.01:
                        u[g][c] = 0.01
                    if bg[g][0] <= 1e-09:
                        bg[g][0] = 1e-09
                    b[g][c] = np.dot(bg[g, :], bc[:, c])

                    dG, dH, dbg = FanoOptimize(A, G, H, u, bg, bc, b, g, c, self.lambda2, self.dataweightControl)

                    G[g, :] = G[g, :] + alpha * dG
                    H[:, c] = H[:, c] + alpha * dH
                    bg[g, :] = bg[g, :] + alpha * dbg
                u = np.dot(G, H)
                b = np.dot(bg, bc)
            alpha = alpha*(1 - float(step/self.steps))
            print("number of iteration: ", step+1, "/", self.steps, LossFuncGre)
        u = np.dot(G, H)
        u[u < 0] = 0

        if Lossf[-1] > Lossf[0]:
            return u, G, H
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
        H = self.H.copy()
        A = self.A.copy()
        u = np.dot(G, H)
        ac = np.ones((1, self.cells))
        if self.calculateIntialNoiseFactor:
            ag = geta(self.A, u)
        else:
            ag = np.ones((self.genes, 1))

        a = np.dot(ag, ac)

        LossFunc = 0
        LossFuncGre = 0
        Lossf = []

        nonZerosInd = np.nonzero(A)
        numNonZeros = nonZerosInd[0].size
        nonZeroElem = np.concatenate((nonZerosInd[0].reshape(numNonZeros, 1), nonZerosInd[1].reshape(numNonZeros, 1)),
                                     axis=1)

        for step in range(self.steps):
            if self.calculateLossFunc:
                for element in nonZeroElem:
                    g = element[0]
                    c = element[1]
                    LossFuncCG = LossFunctionConstantCoefficientVariation(u[g][c], a[g][c], A[g][c], G, H, g, c, self.lambda2, self.dataweightControl, self.verbose)
                    LossFunc = LossFunc + LossFuncCG
            else:
                LossFunc = (step + 1)*self.eps


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
                    g = element[0]
                    c = element[1]
                    if u[g][c] <= 0.01:
                        u[g][c] = 0.01
                    if ag[g][0] <= 1e-09:
                        ag[g][0] = 1e-09
                    if ac[0][c] <= 1e-09:
                        ac[0][c] = 1e-09

                    dG, dH, dag = CCVOptimize(A, G, H, u, ag, ac, a, g, c, self.lambda2, self.dataweightControl)

                    G[g, :] = G[g, :] + self.alpha * dG
                    H[:, c] = H[:, c] + self.alpha * dH
                    ag[g, :] = ag[g, :] + self.alpha * dag
                u = np.dot(G, H)
                a = np.dot(ag, ac)
            print("number of iteration: ", step + 1, "/", self.steps, LossFuncGre)
        u = np.dot(G, H)
        u[u < 0] = 0

        return u, G, H

    def smurf_impute(self, initialDataFrame):
        self.initialDataFrame = initialDataFrame
        self.genes = initialDataFrame.shape[0]
        self.cells = initialDataFrame.shape[1]
        if not self.K:
            self.K = min(self.genes, self.cells)
            self.batchSize = self.K * 10
        self.genesNames = initialDataFrame._stat_axis.values.tolist()
        self.cellsNames = initialDataFrame.columns.values.tolist()

        print("Running SCEnd on {} cells and {} genes, with {} dimension".format(self.cells, self.genes, self.K))

        if self.normalize:
            print("normalizing data by library size...")
            normalizedDataframe, self.size_factors = utiles.dataNormalization(self.initialDataFrame)
            self.A = normalizedDataframe.values

            # if self.genes == self.cells:
            #     energy = 0
            #     u, sigma, vt = np.linalg.svd(self.A)
            #     sigma_square = np.power(sigma, 2)
            #     sigma_square_sum = np.sum(sigma_square)
            #     for i in range(self.genes):
            #         energy += sigma_square[i]
            #         if energy > sigma_square_sum * 0.9:
            #             self.K = i + 1
            #             self.batchSize = self.K * 10

            self.G, self.H = initialMatrices(normalizedDataframe.values, self.K)

            self._check_params()

            print("preprocessing data...")

            if self.noise_model == "CV":
                u, G, H = self.MFConstantVariance()

            if self.noise_model == "Fano":
                u, G, H = self.MFFano()

            if self.noise_model == "CCV":
                u, G, H = self.MFConstCoeffiVariation()

            newDataFrame = pd.DataFrame(u, index=self.genesNames, columns=self.cellsNames)
            newDataFrame = newDataFrame * self.size_factors

            res = {}

            res["estimate"] = newDataFrame
            res["gene latent factor matrix"] = pd.DataFrame(G, index=self.genesNames, columns=None)
            res["cell latent factor matrix"] = pd.DataFrame(H, index=None, columns=self.cellsNames)

            self.glfm = G
            self.clfm = H

            if self.estmate_only:
                return res["estimate"], G, H
            else:
                return res

        else:
            self.A = self.initialDataFrame.values
            self.G, self.H = initialMatrices(self.initialDataFrame.values, self.K)

            self._check_params()

            print("preprocessing data...")

            if self.noise_model == "CV":
                u, G, H = self.MFConstantVariance()

            if self.noise_model == "Fano":
                u, G, H = self.MFFano()

            if self.noise_model == "CCV":
                u, G, H = self.MFConstCoeffiVariation()

            newDataFrame = pd.DataFrame(u, index=self.genesNames, columns=self.cellsNames)

            res = {}

            res["estimate"] = newDataFrame
            res["gene latent factor matrix"] = pd.DataFrame(G, index=self.genesNames, columns=None)
            res["cell latent factor matrix"] = pd.DataFrame(H, index=None, columns=self.cellsNames)

            self.glfm = G
            self.clfm = H


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




























