#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commands, class, and functions from external modules.
"""

# NUMPY COMMANDS

from numpy import ndarray                # class of ndarray.

# element-wise operations on matrix
from numpy import abs as absolute        # absolute values of matrix elements.
from numpy import power as power         # power of matrix elements.
from numpy import sqrt as sqrt           # square root of matrix elements.
from numpy import real as real           # real part of a matrix.
from numpy import cos as cos             # elementwise cos of a matrix.
from numpy import sin as sin             # elementwise sin of a matrix.
from numpy import tan as tan             # elementwise sin of a matrix.
from numpy import log as log             # elementwise neperian logarithm.
from numpy import exp as exp             # elementwise exponential.
from numpy import log2 as log2           # elementwise logarithm in base 2.
from numpy import floor
from numpy import ceil
from numpy import sign

# global operations on matrix.
from numpy import amin as amin           # find the minimum of a matrix.
from numpy import amax as amax           # find the minimum of a matrix.
from numpy import argmin                 # find the minimum argument.
from numpy import sum as sum             # compute sums.
from numpy import mean as mean           # compute mean.
from numpy import median as median       # compute median.
from numpy import std as std             # compute std.
from numpy import cumsum as cumsum       # compute cumulated sums.
from numpy import diff as diff           # compute successive differences.
from numpy import sort as sort           # sort values of a matrix.
from numpy import nonzero as nonzero     # find elements checking a condition.
from numpy import unique as unique       # select different values of a matrix.
from numpy import fmax

# matrix manipulation
from numpy.matlib import repmat          # replication of a matrix.
from numpy import reshape                # reshape a matrix.
from numpy import array                  # create an array.
from numpy import asmatrix               # interpret as a matrix.

# matrix creation
from numpy import zeros as zeros         # create a matrix with 0.
from numpy import ones as ones           # create a matrix with 1.
from numpy import eye as eye             # create the identity matrix.
from numpy import linspace as linspace   # uniform sampling between two values.
from numpy import arange as arange       # uniform discrete sampling.

# matrix operation
from numpy import concatenate            # concatenate matrices.
from numpy import transpose as transpose  # transpose of a matrix.
from numpy import fliplr as fliplr       # switch a matrix from left to right.
from numpy import matmul                 # matrix multiplication.
from numpy import multiply as multiply   # element wise matrix multiplication.
from numpy import mod, divmod
from numpy import floor_divide

# fft
from numpy import fft                    # fft tools.

# random variables
from numpy.random import randn as randn  # generate standard normal variables.
from numpy.random import rand as rand    # generate uniform variables.
from numpy.random import randint         # generate integer uniform variables.
from numpy.random import permutation     # generate random permutation.
from numpy.random import seed, get_state, set_state

# MATPLOTLIB
# graphical tools
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Storage tools
import pickle

# MATH FUNCTIONS AND CONSTANTS
from math import atan2
from math import pi
from numpy import inf

from scipy.special import gamma   # Gamma function.
from scipy.special import beta    # Beta function.
from scipy.special import betainc  # Incomplete Beta function.

# Optimization function
from scipy.optimize import minimize, least_squares
from numpy.linalg import solve


# Function to approximate useful integrals
def BETA_H(coord, alp1, alp2, H):

    pi2 = pi / 2
    H = H + 0.5

    theta = zeros((coord.ncoord, 1))
    for i in range(0, theta.size):
        if coord.xy[i, 0] != 0:
            theta[i, 0] = atan2(coord.xy[i, 1], coord.xy[i, 0])
        else:
            theta[i, 0] = sign(coord.xy[i, 1]) * pi2
    s1 = sin(alp1 - theta) / 2
    s2 = sin(alp2 - theta) / 2

    G = zeros((coord.ncoord, 1))
    for i in range(0, coord.ncoord):
        if ((alp1 - pi2 <= theta[i, 0]) & (theta[i, 0] <= alp2 - pi2)):
            G[i, 0] = betainc(H, H, 0.5 - s2[i, 0]) +\
                betainc(H, H, 0.5 - s1[i, 0])
        else:
            if ((alp1 + pi2 <= theta[i, 0]) & (theta[i, 0] <= alp2 + pi2)):
                G[i, 0] = betainc(H, H, 0.5 + s2[i, 0]) +\
                    betainc(H, H, 0.5 + s1[i, 0])
            else:
                G[i, 0] = abs(betainc(H, H, 0.5 - s2[i, 0]) -
                              betainc(H, H, 0.5 - s1[i, 0]))

    G = G * beta(H, H)
    return G


def Kval(H):
    """
    Compute integrals for a value useful for the computation of variograms.
    """
    val = pi / (H * gamma(2 * H) * sin(H * pi))
    return val
