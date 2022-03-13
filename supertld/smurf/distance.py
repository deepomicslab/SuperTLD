# _*_ coding: utf-8 _*_
"""
Time:     2021/8/27 21:32
Author:   WANG Bingchen
Version:  V 0.1
File:     distance.py
Describe: 
"""

import numba
import numpy as np


@numba.njit()
def adjustment(x, k, a):
    return np.exp(k*a*np.cos(x))


@numba.njit()
def adjustmentGrad(x, k, a):
    return -k*a*np.exp(k*a*np.cos(x))*np.sin(x)



@numba.njit(fastmath=True)
def distanceInOval(x, y, a=3, b=2, k=0.2):
    """

    :param x: high-dimension embedding of cell A
    :param y: high-dimension embedding of cell B
    :param a: major axis length
    :param b: minor axie length
    :param k: Deformation parameter

    :return: distance between cell A and B in oval whose function is

                    x^2/a^2 + y^2/(t(x)*b^2) = 1

                    where t(x) = exp(kx)
    """

    result = 0.0

    x = x % (2*np.pi)
    y = y % (2*np.pi)
    for i in range(x.shape[0]):
        result += (a*np.cos(x[i]) - a*np.cos(y[i]))**2 + (b*np.sqrt(adjustment(x[i], k, a))*np.sin(x[i]) - b*np.sqrt(adjustment(y[i], k, a))*np.sin(y[i]))**2
    d = np.sqrt(result)
    grad1 = -(a**2)*(np.cos(x) - np.cos(y))*np.sin(x)
    grad2 = (b**2)*(np.sqrt(adjustment(x, k, a))*np.sin(x) - np.sqrt(adjustment(y, k, a))*np.sin(y))
    grad3 = adjustmentGrad(x, k, a)*np.sin(x)/(2*np.sqrt(adjustment(x, k, a))) + np.cos(x)*np.sqrt(adjustment(x, k, a))
    grad = grad1 + grad2 * grad3

    grad = grad/(d + 1e-6)

    return d, grad
