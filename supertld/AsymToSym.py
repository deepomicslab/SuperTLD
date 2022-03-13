# _*_ coding: utf-8 _*_
"""
Time:     2021/12/14 22:50
Author:   ZHANG Yuwei
Version:  V 0.1
File:     AsymToSym.py
Describe:
"""
import numpy as np


def asym_to_sym(matrix, mode="col", extraNorm=False, hic=None, alpha=1):
    if mode == "row":
        if hic != None:
            hic_matrix = acquire_hic(hic)
            if len(matrix) != len(hic_matrix):
                raise ValueError("RNA-associated data and hic's row length doesn't match.")
            # matrix = np.append(np.sqrt(alpha) * min_max_norm(matrix), np.sqrt(1-alpha)*hic_matrix, axis=1)
            matrix = np.append(np.sqrt(alpha) * matrix, np.sqrt(1 - alpha) * hic_matrix, axis=1)
        oldSum = np.sum(matrix)  # sum of the input matrix
        print("Matrix sum for original matrix: {}".format(oldSum))
        matrix = np.dot(matrix, matrix.T)
    elif mode == "col":
        if hic != None:
            hic_matrix = acquire_hic(hic)
            if len(matrix[0]) != len(hic_matrix):
                raise ValueError("RNA-associated data and hic's column length doesn't match.")
            # matrix = np.append(np.sqrt(alpha) * min_max_norm(matrix), np.sqrt(1-alpha)*hic_matrix.T, axis=0)
            matrix = np.append(np.sqrt(alpha) * matrix, np.sqrt(1 - alpha) * hic_matrix.T, axis=0)
        oldSum = np.sum(matrix)  # sum of the input matrix
        print("Matrix sum for original matrix: {}".format(oldSum))
        matrix = np.dot(matrix.T, matrix)
    else:
        raise ValueError("the setting of mode do not valid")
    if extraNorm:
        norm_matrix = BisectionNorm(matrix, oldSum).norm()
        try:
            if norm_matrix == None:
                print("apply the power to the singular values of symmetric matrix")
                norm_matrix = svd_process(matrix, power=0.5)
        except:
            pass
        matrix = norm_matrix
    else:
        matrix = svd_process(matrix, power=1)
    return matrix


def acquire_hic(hic_path):
    hic_sym = np.loadtxt(hic_path)
    # hic_sym = min_max_norm(hic_sym)
    values, vectors = np.linalg.eigh(hic_sym)   # increasing order
    dim = len(np.where(values > 0)[0])
    values = values[-dim:]
    vectors = vectors[:, -dim:]
    X = np.dot(vectors, np.diag(np.sqrt(values)))  # hic = X.T * X
    print("hic recomb error: {} with shape {}".format(np.sum(hic_sym - np.dot(X, X.T)), X.shape))
    return X


class BisectionNorm():
    def __init__(self, matrix, oldSum):
        self.matrix = matrix
        self.length = len(self.matrix)
        self.oldSum = oldSum
        self.u = None
        self.vt = None
        self.sigma = None

    def norm(self):
        for i in range(self.length):
            self.matrix[i, i] = 0
        # print(np.sum(self.matrix))
        self.u, self.sigma, self.vt = np.linalg.svd(self.matrix)
        alpha = self.cal_val(1e-6, 10, 1e-6)
        print(alpha, "--------------")
        if alpha == None:
            return None
        else:
            new_matrix = np.dot(np.dot(self.u, np.diag(np.power(self.sigma, alpha))), self.vt)
            new_matrix[new_matrix < 0] = 0
            # print(np.sum(new_matrix), oldSum)
            return new_matrix

    def cal_val(self, start, end, precision):
        s = self.func(start)
        e = self.func(end)
        if e * s > 0:
            print("no solution", self.oldSum)
            print("s = {}, e = {}".format(s, e))
            return None
        elif s == 0:
            print("Solution is ", start)
            return start
        elif e == 0:
            print("Solution is ", end)
            return end
        else:
            while abs(end - start) > precision:
                mid = (start + end) / 2.0
                m = self.func(mid)
                if m == 0:
                    print("Solution is ", mid)
                    return mid
                elif s * m < 0:
                    end = mid
                elif m * e < 0:
                    start = mid
            print(start, mid, end)
            return start

    def func(self, alpha):
        matrix = np.dot(np.dot(self.u, np.diag(np.power(self.sigma, alpha))), self.vt)
        for i in range(len(matrix)):
            matrix[i, i] = 0
        matrix[matrix < 0] = 0
        result = np.sum(matrix) - self.oldSum
        # print(alpha, np.sum(matrix), oldSum, result)
        return result


def svd_process(input_matrix, power=1):
    u, sigma, vt = np.linalg.svd(input_matrix)
    length = len(input_matrix)
    tmp_sigma = np.zeros(length)
    for i in range(length):
        if power == -1:
            if (sigma[i] >= 1e-6):
                tmp_sigma[i] = pow(sigma[i], power)
        else:
            try:
                tmp_sigma[i] = pow(sigma[i], power)
            except:
                print("There's no {}th singular value of input matrix.".format(i))
    if power == -1:
        tmp_sigma.sort(reverse=True)

    new_matrix = np.dot(np.dot(u, np.diag(tmp_sigma)), vt)
    new_matrix[np.where(new_matrix < 0)] = 0  # remove negative values from new matrix
    return new_matrix
