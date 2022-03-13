# _*_ coding: utf-8 _*_
"""
Time:     2021/7/22 17:34
Author:   WANG Bingchen, ZHANG Yuwei
Version:  V 0.1
File:     optimize.py
Describe:
"""


from scipy.special import psi
from numpy import log
import numpy as np


def CVOptimize(A, G, H, u, Vg, Vc, v, g, c, lambda2, weight):
    DFG1 = (2 * u[g][c] / v[g][c]) * H[:, c] * log(u[g][c]) + u[g][c] * H[:, c] / v[g][c]
    DFG2 = -2 * u[g][c] * log(v[g][c]) * H[:, c] / v[g][c]
    DFG3 = -psi((u[g][c] ** 2) / v[g][c]) * 2 * u[g][c] * H[:, c] / v[g][c]
    DFG4 = psi(A[g][c] + ((u[g][c] ** 2) / v[g][c])) * 2 * u[g][c] * H[:, c] / v[g][c]
    DFG5 = -(2 * u[g][c] * H[:, c] / v[g][c]) * log(1 + u[g][c] / v[g][c]) - (
            A[g][c] + u[g][c] ** 2 / v[g][c]) * H[:, c] / (v[g][c] + u[g][c])
    DFG6 = - 2*lambda2 * H[:, c]
    DFG = DFG1 + DFG2 + DFG3 + DFG4 + DFG5 + DFG6

    DFH1 = (2 * u[g][c] / v[g][c]) * G[g, :] * log(u[g][c]) + u[g][c] * G[g, :] / v[g][c]
    DFH2 = -2 * u[g][c] * log(v[g][c]) * G[g, :] / v[g][c]
    DFH3 = -psi((u[g][c] ** 2) / v[g][c]) * 2 * u[g][c] * G[g, :] / v[g][c]
    DFH4 = psi(A[g][c] + u[g][c] ** 2 / v[g][c]) * 2 * u[g][c] * G[g, :] / v[g][c]
    DFH5 = -(2 * u[g][c] * G[g, :] / v[g][c]) * log(1 + u[g][c] / v[g][c]) - (
            A[g][c] + u[g][c] ** 2 / v[g][c]) * G[g, :] / (v[g][c] + u[g][c])
    DFH6 = - 2*lambda2 * G[g, :]
    DFH = DFH1 + DFH2 + DFH3 + DFH4 + DFH5 + DFH6

    DFVg1 = -(u[g][c] ** 2) * log(u[g][c]) / Vg[g][0] ** 2
    DFVg2 = (u[g][c] ** 2) * log(Vg[g][0]) / (Vg[g][0] ** 2) - (u[g][c] ** 2) / (Vg[g][0] ** 2)
    DFVg3 = psi((u[g][c] ** 2) / Vg[g][0]) * ((u[g][c] ** 2) / Vg[g][0] ** 2)
    DFVg4 = -psi(A[g][c] + u[g][c] ** 2 / Vg[g][0]) * ((u[g][c] ** 2) / Vg[g][0] ** 2)
    DFVg5 = ((u[g][c] ** 2) / (Vg[g][0] ** 2)) * log(1 + u[g][c] / Vg[g][0]) + (
            A[g][c] + u[g][c] ** 2 / Vg[g][0]) * (u[g][c] / (Vg[g][0] ** 2 + u[g][c] * Vg[g][0]))
    DFVg = DFVg1 + DFVg2 + DFVg3 + DFVg4 + DFVg5

    return DFG * pow(A[g][c], weight), DFH * pow(A[g][c], weight), DFVg * pow(A[g][c], weight)


def FanoOptimize(A, G, H, u, bg, bc, b, g, c, lambda2, weight):
    DFG1 = -H[:, c] / b[g][c] * log(b[g][c])
    DFG2 = -psi(u[g][c] / b[g][c]) * H[:, c] / b[g][c]
    DFG3 = psi(A[g][c] + u[g][c] / b[g][c]) * H[:, c] / b[g][c]
    DFG4 = -(1 / b[g][c]) * log(1 + 1 / b[g][c]) * H[:, c]
    DFG5 = - 2*lambda2 * H[:, c]
    DFG = DFG1 + DFG2 + DFG3 + DFG4 + DFG5

    DFH1 = -G[g, :] / b[g][c] * log(b[g][c])
    DFH2 = -psi(u[g][c] / b[g][c]) * G[g][:] / b[g][c]
    DFH3 = psi(A[g][c] + u[g][c] / b[g][c]) * G[g][:] / b[g][c]
    DFH4 = -(1 / b[g][c]) * log(1 + 1 / b[g][c]) * G[g, :]
    DFH5 = - 2*lambda2 * G[g, :]
    DFH = DFH1 + DFH2 + DFH3 + DFH4 + DFH5

    DFBg1 = u[g][c] / (bg[g][0] ** 2) * (log(bg[g][0]) - 1)
    DFBg2 = psi(u[g][c] / bg[g][0]) * u[g][c] / (bg[g][0] ** 2)
    DFBg3 = -psi(A[g][c] + u[g][c] / bg[g][0]) * u[g][c] / (bg[g][0] ** 2)
    DFBg4 = u[g][c] * log(1 + 1 / bg[g][0]) / (bg[g][0] ** 2) + (A[g][c] + u[g][c]/bg[g][0]) * (1/
            (bg[g][0] + bg[g][0] ** 2))
    DFBg = DFBg1 + DFBg2 + DFBg3 + DFBg4

    return DFG * pow(A[g][c], weight), DFH * pow(A[g][c], weight), DFBg * pow(A[g][c], weight)


def CCVOptimize(A, G, H, u, ag, ac, a, g, c, lambda2, weight):
    DFG1 = -H[:, c] / (a[g][c] * u[g][c])
    DFG2 = (A[g][c] + 1 / a[g][c]) * (H[:, c] / (u[g][c] * (a[g][c] * u[g][c] + 1)))
    DFG3 = - 2*lambda2 * H[:, c]
    DFG = DFG1 + DFG2 + DFG3

    DFH1 = -G[g, :] / (a[g][c] * u[g][c])
    DFH2 = (A[g][c] + 1 / a[g][c]) * (G[g, :] / (u[g][c] * (a[g][c] * u[g][c] + 1)))
    DFH3 = - 2*lambda2 * G[g, :]
    DFH = DFH1 + DFH2 + DFH3

    DFAg1 = (log(ag[g][0]) - 1) / (ag[g][0] ** 2)
    DFAg2 = log(u[g][c]) / (ag[g][0] ** 2)
    DFAg3 = psi(1 / ag[g][0]) / (ag[g][0] ** 2)
    DFAg4 = -psi(A[g][c] + (1 / ag[g][0])) / (ag[g][0] ** 2)
    DFAg5 = log(1 + 1 / (ag[g][0] * u[g][c])) / (ag[g][0] ** 2)
    DFAg6 = (A[g][c] + 1/ag[g][0]) * (1 / (ag[g][0] * u[g][c] ** 2 + u[g][c]))
    DFAg = DFAg1 + DFAg2 + DFAg3 + DFAg4 + DFAg5 + DFAg6

    return DFG * pow(A[g][c], weight), DFH * pow(A[g][c], weight), DFAg * pow(A[g][c], weight)


def CVOptimize_sym(A, G, u, Vb, v, b1, b2, lambda2, weight):

    DFG1 = (2 * u[b1][b2] / v[b1][b2]) * G[b1, :] * log(u[b1][b2]) + u[b1][b2] * G[b1, :] / v[b1][b2]
    DFG2 = -2 * u[b1][b2] * log(v[b1][b2]) * G[b1, :] / v[b1][b2]
    DFG3 = -psi((u[b1][b2] ** 2) / v[b1][b2]) * 2 * u[b1][b2] * G[b1, :] / v[b1][b2]
    DFG4 = psi(A[b1][b2] + ((u[b1][b2] ** 2) / v[b1][b2])) * 2 * u[b1][b2] * G[b1, :] / v[b1][b2]
    DFG5 = -(2 * u[b1][b2] * G[b1, :] / v[b1][b2]) * log(1 + u[b1][b2] / v[b1][b2]) - (
            A[b1][b2] + u[b1][b2] ** 2 / v[b1][b2]) * G[b1, :] / (v[b1][b2] + u[b1][b2])
    DFG6 = - 2*lambda2 * G[b1, :]
    DFG_b2 = DFG1 + DFG2 + DFG3 + DFG4 + DFG5 + DFG6

    if b1 == b2:
        DFG_b1 = - 2 * lambda2 * G[b2, :]
    else:
        DFH1 = (2 * u[b1][b2] / v[b1][b2]) * G[b2, :] * log(u[b1][b2]) + u[b1][b2] * G[b2, :] / v[b1][b2]
        DFH2 = -2 * u[b1][b2] * log(v[b1][b2]) * G[b2, :] / v[b1][b2]
        DFH3 = -psi((u[b1][b2] ** 2) / v[b1][b2]) * 2 * u[b1][b2] * G[b2, :] / v[b1][b2]
        DFH4 = psi(A[b1][b2] + ((u[b1][b2] ** 2) / v[b1][b2])) * 2 * u[b1][b2] * G[b2, :] / v[b1][b2]
        DFH5 = -(2 * u[b1][b2] * G[b2, :] / v[b1][b2]) * log(1 + u[b1][b2] / v[b1][b2]) - (
                A[b1][b2] + u[b1][b2] ** 2 / v[b1][b2]) * G[b2, :] / (v[b1][b2] + u[b1][b2])
        DFH6 = - 2*lambda2 * G[b2, :]
        DFG_b1 = DFH1 + DFH2 + DFH3 + DFH4 + DFH5 + DFH6

    pVpv1 = Vb[b2][0] / (2 * v[b1][b2])
    DFVb1 = -((u[b1][b2] ** 2) * log(u[b1][b2]) * pVpv1) / (v[b1][b2] ** 2)
    DFVb2 = ((u[b1][b2] ** 2) * (log(v[b1][b2])-1) * pVpv1) / (v[b1][b2] ** 2)
    DFVb3 = psi((u[b1][b2] ** 2) / v[b1][b2]) * ((u[b1][b2] ** 2) *pVpv1 / (v[b1][b2] ** 2))
    DFVb4 = -psi(A[b1][b2] + (u[b1][b2] ** 2 / v[b1][b2])) * ((u[b1][b2] ** 2) * pVpv1 / (v[b1][b2] ** 2))
    DFVb5 = ((u[b1][b2] ** 2) * pVpv1/ (v[b1][b2] ** 2)) * log(1 + u[b1][b2] / v[b1][b2]) + (
            A[b1][b2] + (u[b1][b2] ** 2 / v[b1][b2])) * (u[b1][b2] * pVpv1 / (v[b1][b2] ** 2 + u[b1][b2] * v[b1][b2]))
    DFV_b1 = DFVb1 + DFVb2 + DFVb3 + DFVb4 + DFVb5

    if b1 == b2:
        DFV_b2 = 0
    else:
        pVpv2 = Vb[b1][0] / (2 * v[b1][b2])
        DFVh1 = -(u[b1][b2] ** 2) * log(u[b1][b2]) * pVpv2 / (v[b1][b2] ** 2)
        DFVh2 = (u[b1][b2] ** 2) * (log(v[b1][b2])-1) * pVpv2 / (v[b1][b2] ** 2)
        DFVh3 = psi((u[b1][b2] ** 2) / v[b1][b2]) * ((u[b1][b2] ** 2) * pVpv2 / (v[b1][b2] ** 2))
        DFVh4 = -psi(A[b1][b2] + (u[b1][b2] ** 2 / v[b1][b2])) * ((u[b1][b2] ** 2) * pVpv2 / (v[b1][b2] ** 2))
        DFVh5 = ((u[b1][b2] ** 2) * pVpv2 / (v[b1][b2] ** 2)) * log(1 + u[b1][b2] / v[b1][b2]) + (
                A[b1][b2] + u[b1][b2] ** 2 / v[b1][b2]) * (u[b1][b2] * pVpv2 / (v[b1][b2] ** 2 + u[b1][b2] * v[b1][b2]))
        DFV_b2 = DFVh1 + DFVh2 + DFVh3 + DFVh4 + DFVh5

    return DFG_b1 * pow(A[b1][b2], weight), DFG_b2 * pow(A[b1][b2], weight), DFV_b1 * pow(A[b1][b2], weight), DFV_b2 * pow(A[b1][b2], weight)


def FanoOptimize_sym(A, G, u, bg, b, b1, b2, lambda2, weight):
    DFG1 = -(G[b1, :] / b[b1][b2]) * log(b[b1][b2])
    DFG2 = -psi(u[b1][b2] / b[b1][b2]) * G[b1, :] / b[b1][b2]
    DFG3 = psi(A[b1][b2] + u[b1][b2] / b[b1][b2]) * G[b1, :] / b[b1][b2]
    DFG4 = -(1 / b[b1][b2]) * log(1 + 1 / b[b1][b2]) * G[b1, :]
    DFG5 = -2*lambda2 * G[b1, :]
    DFG_b2 = DFG1 + DFG2 + DFG3 + DFG4 + DFG5

    if b1 == b2:
        DFG_b1 = - 2 * lambda2 * G[b2, :]
    else:
        DFH1 = -(G[b2, :] / b[b1][b2]) * log(b[b1][b2])
        DFH2 = -psi(u[b1][b2] / b[b1][b2]) * G[b2, :] / b[b1][b2]
        DFH3 = psi(A[b1][b2] + u[b1][b2] / b[b1][b2]) * G[b2, :] / b[b1][b2]
        DFH4 = -(1 / b[b1][b2]) * log(1 + 1 / b[b1][b2]) * G[b2, :]
        DFH5 = -2*lambda2 * G[b2, :]
        DFG_b1 = DFH1 + DFH2 + DFH3 + DFH4 + DFH5

    pBpb1 = bg[b2][0] / (2 * b[b1][b2])
    DFBg1 = (u[b1][b2] / (b[b1][b2] ** 2)) * (log(b[b1][b2]) - 1) * pBpb1
    DFBg2 = psi(u[b1][b2] / b[b1][b2]) * u[b1][b2] * pBpb1/ (b[b1][b2] ** 2)
    DFBg3 = -psi(A[b1][b2] + (u[b1][b2] / b[b1][b2])) * u[b1][b2] * pBpb1/ (b[b1][b2] ** 2)
    DFBg4 = u[b1][b2] * log(1 + 1 / b[b1][b2]) * pBpb1 / (b[b1][b2] ** 2) + (
            A[b1][b2] + u[b1][b2]/b[b1][b2]) * pBpb1 / (b[b1][b2] + (b[b1][b2] ** 2))
    DFB_b1 = DFBg1 + DFBg2 + DFBg3 + DFBg4

    if b1 == b2:
        DFB_b2 = 0
    else:
        pBpb2 = bg[b1][0] / (2 * b[b1][b2])
        DFBh1 = (u[b1][b2] / (b[b1][b2] ** 2)) * (log(b[b1][b2]) - 1) * pBpb2
        DFBh2 = psi(u[b1][b2] / b[b1][b2]) * u[b1][b2] * pBpb2 / (b[b1][b2] ** 2)
        DFBh3 = -psi(A[b1][b2] + (u[b1][b2] / b[b1][b2])) * u[b1][b2] * pBpb2 / (b[b1][b2] ** 2)
        DFBh4 = u[b1][b2] * log(1 + 1 / b[b1][b2]) * pBpb2 / (b[b1][b2] ** 2) + (
                    A[b1][b2] + u[b1][b2] / b[b1][b2]) * pBpb2 / (b[b1][b2] + (b[b1][b2] ** 2))
        DFB_b2 = DFBh1 + DFBh2 + DFBh3 + DFBh4


    #return DFG_b1*A[b1][b2], DFG_b2*A[b1][b2], DFB_b1*A[b1][b2], DFB_b2*A[b1][b2]
    return DFG_b1 * pow(A[b1][b2], weight), DFG_b2 * pow(A[b1][b2], weight), DFB_b1 * pow(A[b1][b2], weight), DFB_b2 * pow(A[b1][b2], weight)


def CCVOptimize_sym(A, G, u, ag, a, b1, b2, lambda2, weight):
    DFG1 = - G[b1, :] / (a[b1][b2] * u[b1][b2])
    DFG2 = (A[b1][b2] + 1 / a[b1][b2]) * G[b1, :] / (u[b1][b2] * (a[b1][b2] * u[b1][b2] + 1))
    DFG3 = - 2*lambda2 * G[b1, :]
    DFG_b2 = DFG1 + DFG2 + DFG3

    if b1 == b2:
        DFG_b1 = - 2* lambda2 * G[b2, :]
    else:
        DFH1 = - G[b2, :] / (a[b1][b2] * u[b1][b2])
        DFH2 = (A[b1][b2] + 1 / a[b1][b2]) * G[b2, :] / (u[b1][b2] * (a[b1][b2] * u[b1][b2] + 1))
        DFH3 = - 2*lambda2 * G[b2, :]
        DFG_b1 = DFH1 + DFH2 + DFH3

    pApa1 = ag[b2][0] / (2 * a[b1][b2])
    DFAg1 = (log(a[b1][b2]) - 1) * pApa1 / (a[b1][b2] ** 2)
    DFAg2 = log(u[b1][b2]) * pApa1 / (a[b1][b2] ** 2)
    DFAg3 = psi(1 / a[b1][b2]) * pApa1/ (a[b1][b2] ** 2)
    DFAg4 = -psi(A[b1][b2] + (1 / a[b1][b2])) * pApa1/ (a[b1][b2] ** 2)
    DFAg5 = log(1 + 1 / (a[b1][b2] * u[b1][b2])) * pApa1/ (a[b1][b2] ** 2)
    DFAg6 = (A[b1][b2] + 1 / a[b1][b2]) * pApa1 /( a[b1][b2] * (u[b1][b2] ** 2) + u[b1][b2])
    DFA_b1 = DFAg1 + DFAg2 + DFAg3 + DFAg4 + DFAg5 + DFAg6

    if b1 == b2:
        DFA_b2 = 0
    else:
        pApa2 = ag[b1][0] / (2 * a[b1][b2])
        DFAh1 = (log(a[b1][b2]) - 1) * pApa2 / (a[b1][b2] ** 2)
        DFAh2 = log(u[b1][b2]) * pApa2 / (a[b1][b2] ** 2)
        DFAh3 = psi(1 / a[b1][b2]) * pApa2 / (a[b1][b2] ** 2)
        DFAh4 = -psi(A[b1][b2] + (1 / a[b1][b2])) * pApa2 / (a[b1][b2] ** 2)
        DFAh5 = log(1 + 1 / (a[b1][b2] * u[b1][b2])) * pApa2 / (a[b1][b2] ** 2)
        DFAh6 = (A[b1][b2] + 1 / a[b1][b2]) * pApa2 /( a[b1][b2] * (u[b1][b2] ** 2) + u[b1][b2])
        DFA_b2 = DFAh1 + DFAh2 + DFAh3 + DFAh4 + DFAh5 + DFAh6

    return DFG_b1 * pow(A[b1][b2], weight), DFG_b2 * pow(A[b1][b2], weight), DFA_b1 * pow(A[b1][b2], weight), DFA_b2 * pow(A[b1][b2], weight)
