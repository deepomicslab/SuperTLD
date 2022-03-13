# _*_ coding: utf-8 _*_
"""
Time:     2021/12/2 22:04
Author:   ZHANG Yuwei
Version:  V 0.2
File:     lossFunc_sym.py
Describe:
"""
from scipy.special import gammaln
from numpy import log
import numpy as np


def LossFunctionConstantVariance(u, v, y, G, b1, b2, lambda2, weight, verbose=False):
    u = max(u, 0.01)
    v = max(v, 1e-09)
    fun1 = ((u ** 2) / v) * log(u)
    fun2 = -((u ** 2) * (1 / v) * log(v))
    fun3 = -gammaln(u ** 2 / v)
    fun4 = gammaln(y + u**2 / v)
    fun5 = -(y + u ** 2 / v) * log(1 + u / v)
    fun6 = - lambda2 * (np.dot(G[b1, :], G[b1, :].transpose()) + np.dot(G[b2, :], G[b2, :].transpose()))    # regularization
    fun = fun1 + fun2 + fun3 + fun4 + fun5 + fun6
    if verbose and b1 % 100 == 0 and b2 % 100 == 0:
        print(fun-fun6, fun6, fun6/lambda2, pow(y, weight))
    return fun * pow(y, weight)


def LossFunctionFano(u, b, y, G, b1, b2, lambda2, weight, verbose=False):
    u = max(u, 0.01)
    b = max(b, 1e-09)
    fun1 = - (u/b) * log(b)
    fun2 = -gammaln(u/b)
    fun3 = gammaln(y + (u/b))
    fun4 = -(y + (u/b))*log(1 + (1/b))
    fun5 = - lambda2 * (np.dot(G[b1, :], G[b1, :].transpose()) + np.dot(G[b2, :], G[b2, :].transpose()))
    fun = fun1 + fun2 + fun3 + fun4 + fun5
    if verbose and b1 % 100 == 0 and b2 % 100 == 0:
        print(fun-fun5, fun5, fun5/lambda2, pow(y, weight))
    # if b1==b2 == 464:
    #     print(u, b, y, np.round(fun1, 2), np.round(fun2,2), np.round(fun3,2), np.round(fun4,2), np.round(fun5,2), np.round(fun,2), b1)
    if np.isnan(fun) or np.isinf(fun):
        exit()
    return fun * pow(y, weight)


def LossFunctionConstantCoefficientVariation(u, a, y, G, b1, b2, lambda2, weight, verbose=False):
    u = max(u, 0.01)
    a = max(a, 1e-09)
    fun1 = -(1/a) * log(a)
    fun2 = -(1/a) * log(u)
    fun3 = -gammaln(1/a)
    fun4 = gammaln(y + 1/a)
    fun5 = -(y + 1/a) * log(1 + 1/(a * u))
    fun6 = - lambda2 * (np.dot(G[b1, :], G[b1, :].transpose()) + np.dot(G[b2, :], G[b2, :].transpose()))
    fun = fun1 + fun2 + fun3 + fun4 + fun5 + fun6
    if verbose and b1 % 100 == 0 and b2 % 100 == 0:
        print(fun-fun6, fun6, fun6/lambda2, pow(y, weight))
    return fun * pow(y, weight)
