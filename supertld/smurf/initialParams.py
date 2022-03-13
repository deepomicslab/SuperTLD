# _*_ coding: utf-8 _*_
"""
Time:     2021/7/22 17:36
Author:   WANG Bingchen
Version:  V 0.1
File:     initialParams.py
Describe: 
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from numpy import log
import warnings


def initialMatrices(U_0, K):
    U = U_0
    u, sigma, vt = np.linalg.svd(U)
    print(sigma[:10])
    u = u[:, :K]
    sigma = sigma[:K]
    vt = vt[:K, :]
    G = np.dot(u, np.diag(sigma))    # U * sigma
    H = vt # V.T

    return G, H

def initialMatrix_sym(U_0, K):
    U = U_0
    values, vectors = np.linalg.eigh(U) # increasing order
    positives = len(np.where(values>0)[0])
    if K > positives:
        # values = np.abs(values)
        K = positives
        print("Change the dimension to {}".format(K))
    order = np.arange(-1, -K - 1, -1)   # take the largest K eigenvalue
    values = values[order]
    vectors = vectors[:, order]
    return np.dot(vectors, np.diag(np.sqrt(values)))    # X where U = X * X.T

def loglikv(args):
    mu, y = args
    fun = lambda x: -(sum((mu**2)*log(mu) / x[0]) -
                      sum((mu**2)*log(x[0]) / x[0]) -
                      sum(gammaln(mu**2 / x[0])) +
                      sum(gammaln(y + mu**2 / x[0])) -
                      sum((y + mu**2 / x[0])*log(1 + mu / x[0])))
    return fun

def loglikb(args):
    mu, y = args
    fun = lambda x: -(sum((mu / x[0]) * log(1 / x[0])) -
                      sum(gammaln(mu / x[0])) +
                      sum(gammaln(y + (mu / x[0]))) -
                      sum((y + (mu / x[0])) * log(1 + (1 / x[0]))))
    return fun


def loglika(args):
    mu, y = args
    n = len(y)
    fun = lambda x: -(n / x[0]*log(1 / x[0]) -
                      sum(1 / x[0]*log(mu)) -
                      n*gammaln(1 / x[0]) +
                      sum(gammaln(y + 1 / x[0])) -
                      sum((y + 1 / x[0])*log(1 + 1 / (x[0]*mu))))
    return fun


def con(args):
    Xmin, Xmax = args
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - Xmin},
            {'type': 'ineq', 'fun': lambda x: Xmax - x[0]})
    return cons


def getv(dataFrame, mu):
    warnings.filterwarnings("ignore")
    (genes, cells) = dataFrame.shape
    vg = []
    for g in range(genes):
        mu_g = mu[g, :]
        Y_g = np.array(dataFrame.iloc[g])
        x0 = np.asarray((1))
        args = (mu_g, Y_g)
        args1 = (1e-07, 2)
        cons = con(args1)
        res = minimize(loglikv(args), x0=x0, constraints=cons)
        if res.success:
            vg.append(res.x[0])
        else:
            vg.append(1)
    vg = np.array(vg)
    vg = vg.reshape(genes, 1)
    return vg


def getb(dataFrame, mu):
    warnings.filterwarnings("ignore")
    (genes, cells) = dataFrame.shape
    bg = []
    for g in range(genes):
        mu_g = mu[g, :]
        Y_g = np.array(dataFrame.iloc[g])
        x0 = np.asarray((1))
        args = (mu_g, Y_g)
        args1 = (1e-07, 2)
        cons = con(args1)
        res = minimize(loglikb(args), x0=x0, constraints=cons)
        if res.success:
            bg.append(res.x[0])
        else:
            bg.append(1)
    bg = np.array(bg)
    bg = bg.reshape(genes, 1)
    return bg


def geta(dataFrame, mu):
    warnings.filterwarnings("ignore")
    (genes, cells) = dataFrame.shape
    ag = []
    for g in range(genes):
        mu_g = mu[g, :]
        Y_g = np.array(dataFrame.iloc[g])
        x0 = np.asarray((1))
        args = (mu_g, Y_g)
        args1 = (1e-07, 2)
        cons = con(args1)
        res = minimize(loglika(args), x0=x0, constraints=cons)
        if res.success:
            ag.append(res.x[0])
        else:
            ag.append(1)
    ag = np.array(ag)
    ag = ag.reshape(genes, 1)
    return ag