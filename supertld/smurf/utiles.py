# _*_ coding: utf-8 _*_
"""
Time:     2021/7/22 17:31
Author:   WANG Bingchen
Version:  V 0.1
File:     utiles.py
Describe:
"""

import numbers


def check_positive(**params):
    """Check that parameters are positive as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] <= 0:
            raise ValueError(
                "Expected {} > 0, got {}".format(p, params[p]))


def check_int(**params):
    """Check that parameters are integers as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Integral):
            raise ValueError(
                "Expected {} integer, got {}".format(p, params[p]))


def check_bool(**params):
    """Check that parameters are bools as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] is not True and params[p] is not False:
            raise ValueError(
                "Expected {} boolean, got {}".format(p, params[p]))


def check_between(v_min, v_max, **params):
    """Checks parameters are in a specified range

    Parameters
    ----------

    v_min : float, minimum allowed value (inclusive)

    v_max : float, maximum allowed value (inclusive)

    params : object
        Named arguments, parameters to be checked

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] < v_min or params[p] > v_max:
            raise ValueError("Expected {} between {} and {}, "
                             "got {}".format(p, v_min, v_max, params[p]))


def check_noise_model(**params):
    for p in params:
        if (params[p] != "CV") and (params[p] != "Fano") and (params[p] != "CCV"):
            raise ValueError("Expected {} be one in CV, Fano, or CCV, got {}".format(p, params[p]))



def dataNormalization(dataFrame):
    """
    Normalization based on library size of each cell/bin.
    Each bin share the same coverage.

    Parameters:

    -----------

    :param dataFrame: origin dataframe

    -----------

    :return: dataframe normalized，sizefactors
    """
    Ngene, Ncell = dataFrame.shape
    librarySize = dataFrame.sum(axis=0) # sum of a column (for a cell)
    meanLibrarySize = librarySize.mean()
    sizeFactors = librarySize / meanLibrarySize # (Ncell, )
    for i in range(Ncell):
        if sizeFactors[i]:
            dataFrame.iloc[:, i] = dataFrame.iloc[:, i] / sizeFactors[i]
    #normalizedDataFrame = dataFrame / sizeFactors
    return dataFrame, sizeFactors


