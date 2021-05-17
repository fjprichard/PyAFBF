#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commands, class, and functions from external modules.
"""

# NUMPY COMMANDS

from numpy import ndarray                # class of ndarray.

# element-wise operations on matrix
from numpy import absolute  # absolute values of matrix elements.
from numpy import power  # power of matrix elements.
from numpy import sqrt  # square root of matrix elements.
from numpy import real # real part of a matrix.
from numpy import cos  # elementwise cos of a matrix.
from numpy import sin  # elementwise sin of a matrix.
from numpy import tan  # elementwise sin of a matrix.
from numpy import arctan2  # elementwise arctan of a matrix.
from numpy import log  # elementwise neperian logarithm.
from numpy import exp  # elementwise exponential.
from numpy import log2  # elementwise logarithm in base 2.
from numpy import floor
from numpy import ceil
from numpy import sign

# global operations on matrix.
from numpy import amin  # find the minimum of a matrix.
from numpy import amax  # find the minimum of a matrix.
from numpy import argmin  # find the minimum argument.
from numpy import sum  # compute sums.
from numpy import mean  # compute mean.
from numpy import median  # compute median.
from numpy import std  # compute std.
from numpy import cumsum  # compute cumulated sums.
from numpy import diff  # compute successive differences.
from numpy import sort  # sort values of a matrix.
from numpy import nonzero  # find elements checking a condition.
from numpy import unique  # select different values of a matrix.
from numpy import fmax

# matrix manipulation
from numpy.matlib import repmat          # replication of a matrix.
from numpy import reshape                # reshape a matrix.
from numpy import array                  # create an array.
from numpy import asmatrix               # interpret as a matrix.

# matrix creation
from numpy import zeros  # create a matrix with 0.
from numpy import ones  # create a matrix with 1.
from numpy import eye  # create the identity matrix.
from numpy import linspace  # uniform sampling between two values.
from numpy import arange  # uniform discrete sampling.

# matrix operation
from numpy import concatenate  # concatenate matrices.
from numpy import transpose  # transpose of a matrix.
from numpy import fliplr  # switch a matrix from left to right.
from numpy import matmul  # matrix multiplication.
from numpy import multiply  # element wise matrix multiplication.
from numpy import mod, divmod
from numpy import floor_divide

# fft
from numpy import fft  # fft tools.

# random variables
from numpy.random import randn  # generate standard normal variables.
from numpy.random import rand  # generate uniform variables.
from numpy.random import randint  # generate integer uniform variables.
from numpy.random import permutation  # generate random permutation.
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

from scipy.special import gamma  # Gamma function.
from scipy.special import beta  # Beta function.
from scipy.special import betainc  # Incomplete Beta function.

# Optimization function
from scipy.optimize import minimize, least_squares
from numpy.linalg import solve

