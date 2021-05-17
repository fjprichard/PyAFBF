#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for the management of pi-periodic functions.
"""
from afbf.utilities import linspace, pi, concatenate, reshape, nonzero
from afbf.utilities import asmatrix, cos, sin
from afbf.utilities import rand, plt, zeros, amin, amax, argmin, sort, floor
from afbf.utilities import ceil, sqrt, power, randn
from afbf.utilities import sum, diff, mod, floor_divide, mean, std, median
from afbf.utilities import absolute, ones, matmul, arange, array, ndarray
from afbf.utilities import pickle


class perfunction:
    r"""
    Handle some pi-periodic positive parametric functions.
    The available function representations are
    - the representation in a Fourier basis,
    - the representation by step functions.

    In the Fourier representation, the function is defined as
    ..math:: $$f(t)=a_0+\sum_{m=1}^M a_m^{(1)} \cos(2mt)+a_m^{(2)} \sin(2mt),$$
             where $a_0$, $a_m^{(k)}$ are real parameters.

    A step function is defined as
    ..math:: $$ f(t) = a_i$ if $t \in [\psi_i, psi_{i+1}),$$ where $a_i$ are
             non-negative parameters, and $\psi_i$ is a set of ordered angles
             in $[\pi/2, \pi/2]$.

    The definition of step functions can be adapted to include smooth
    transitions between intervals where the function is constant.
    """

    def __init__(self, ftype='step-smooth', param=0, fname='noname'):
        """
        Define a positive periodic function with a specified representation.

        Parameters
        ----------
        ftype : str, optional
            A function representation among 'step-constant', 'step-piecewise',
            'step-ridge', 'step-smooth', and 'Fourier'.
            The default is 'step-smooth'.
        param : int, optional
            Number of parameters. The default is 0.
        fname : str, optional
            The name of the function. The default is 'noname'.
        """
        if not isinstance(fname, str) or not isinstance(ftype, str):
            print('perfunction: provide string arguments for fname and ftype')

        self.fname = fname
        self.ftype = 'undefined'
        self.fparam = None
        self.finter = None

        if ftype == 'Fourier':
            self.InitFourierFunction('init', param)
        elif 'step' in ftype:
            self.InitStepFunction(ftype, 'init', param)
        else:
            print("PeriodicFunction.__init__: invalid representation.")
            print("PeriodicFunction.__init__: function not defined.")

        self.t = None
        self.values = None
        self.basis = None

    def CheckValidity(self):
        """
        Check the validity of the periodic function.

        Returns
        -------
        valid: boolean
            True if valid.

        """
        valid = isinstance(self.fparam, ndarray)
        if "step" in self.ftype:
            valid = valid and isinstance(self.finter, ndarray)

        if valid is False:
            print("The periodic function is not properly defined.")

        return(valid)

    def ChangeParameters(self):
        """
        Change parameters of the function at random while keeping a same
        representation.

        Returns
        -------
        self.fparam: paramters updated.
        """
        if self.ftype == 'Fourier':
            self.InitFourierFunction('update')
        if 'step' in self.ftype:
            self.InitStepFunction(self.ftype, 'update')

    def Evaluate(self, t=None):
        """
        Evaluate the function at some positions.

        Parameters
        ----------
        t : numpy.ndarray, optional
          Positions at which the function is computed. If omitted,
          the function is re-evaluated at points of the previous evaluation.
        """
        if self.CheckValidity() is False:
            return(0)

        if t is not None:
            if isinstance(t, ndarray) is False:
                print('pefunction.Evaluate: set parameter t as ndarray.')
                return(0)
            self.t = reshape(t, (1, t.size))
            self.values = zeros((1, self.t.size))
            if 'step' in self.ftype:
                self.ComputeStepBasis()
            if 'Fourier' in self.ftype:
                self.ComputeFourierBasis()

        if self.basis is None:
            print("PeriodicFunctions.Evaluate: give positions t as ndarray.")
        else:
            self.values = matmul(self.fparam, self.basis)

        return(1)

    def Display(self, nfig=1):
        """
        Plot the graph of the function.

        Parameters
        ----------
        nfig : integer, optional
             The index of the figure. The default is 1.
        """
        if self.CheckValidity() is False:
            return(0)

        t = linspace(-pi, pi, 10000)
        self.Evaluate(t)

        plt.figure(nfig)
        plt.plot(self.t.T, self.values.T, "r-")
        plt.xlabel('t')
        plt.title(self.fname)
        plt.axis([- pi, pi, 0, amax(self.values) + 0.01])
        plt.show()
        return(1)

    def ComputeFeatures(self, m=10000):
        """
        Compute some features describing the function.

        Parameters
        ----------
        m : int, optional
            Number of discrete positions to evaluate the function.
            The default is 10000.

        Returns
        -------
        delta:
            Precision.
        self.stats:
            Basics statistics: min, max, mean, median of the function.
        self.dev:
            Measures of deviations of the function: standard deviation,
            absolute deviations to the mean and the median.
        self.sharpness:
            Measures of sharpness computed from discrete function derivative
            (discrete tv-norm, maximum of absolute discrete derivative).
        """
        if self.CheckValidity() is False:
            return(0)

        t = linspace(- pi / 2, pi / 2, m)
        self.Evaluate(t)
        delta = pi / m
        [stats, dev, sharpness] =\
            DiscreteFunctionDescription(self.values, delta)
        self.stats = stats
        self.dev = dev
        self.sharpness = sharpness

        return delta

    def Save(self, filename):
        """
        Save the periodic function.

        Parameters
        ----------
        filename : str
            File name (without the extension).
        """
        with open(filename + '.pickle', 'wb') as f:
            if 'step' in self.ftype:
                pickle.dump([self.ftype, self.fname, self.fparam, self.finter],
                            f)
            elif 'Fourier' in self.ftype:
                pickle.dump([self.ftype, self.fname, self.fparam], f)

# Functions for the management of the representation by step functions.

    def InitStepFunction(self, ftype='step-constant', mode='init', M=3):
        """
        Define or update a step function at random.

        Parameters
        ----------
        ftype : str, optional
            The type of step function.
            ('step','step-constant','step-ridge','step-smooth').
            The default is 'step-constant'.
        mode : str, optional
            The utilisation mode ('init', 'update').
            The default is 'init'.
        M : int, optional
            Number of steps. The default is 3.

        Returns
        -------
        self.fparam, self.finter
        """
        if 'init' in mode:
            """
            Set the representation.
            """
            self.ftype = 'step-pwise'
            # Set the number of intervals.
            if ('constant' in ftype):
                M = 1
                self.ftype = 'step-constant'
            else:
                if M <= 0:
                    M = 3
                else:
                    M = int(ceil(M))

            if ('ridge' in ftype):
                M = M * 2
                self.ftype = 'step-ridge'

            if ('smooth' in ftype):
                M2 = 2 * M
                self.steptrans = True
                self.ftype = 'step-smooth'
            else:
                M2 = M
                self.steptrans = False

            self.fparam = zeros((1, M))
            self.finter = zeros((1, M2))

            # Set transition intervals for smooth and ridge step functions.
            if ('ridge' in ftype) or ('smooth' in ftype):
                if rand() > 0.5:
                    self.trans = 0
                else:
                    self.trans = 1

            # Set the default method for simulation of step constants.
            self.SetStepSampleMode()

        # Sample interval bounds and constants of steps.
        k = self.SampleStepIntervals()
        self.SampleStepConstants(k)

        if ('ridge' in self.ftype):
            self.fparam[0, arange(self.trans, self.fparam.size, 2)] = 0

    def SampleStepIntervals(self):
        """
        Sample interval bounds of a step function.
        The simulation of the step constants depends on smode.
        - smode[3]='unif': the bounds are uniformly sampled over the interval
        (-pi/2, pi/2).
        - smode[3]='nonunif': the bounds are sampled so that the size of one
        of them is uniformly sampled.
        smode[4] is a minimal interval size.

        Returns
        -------
        k : int.
            The index of the interval with a uniformly sampled size.

        """
        if "step" not in self.ftype:
            print("pefunction.SampleSetIntervals: only for step functions.")
            return(0)

        mode = self.smode[3]
        dmin = self.smode[4]
        ninter = self.finter.size

        if mode == 'unif':
            self.smode[3] = 'unif'
            # Sample interval boundaries.
            finter = sort(rand(1, ninter) * pi - pi / 2, axis=1)
            k = -1
        else:
            self.smode[3] = 'nonunif'
            dmax = pi - dmin
            if dmax <= 0:
                dmax = pi
                dmin = 0
                self.smode[4] = 0

            finter = zeros((1, ninter))
            finter[0, 0] = 0
            # Sample an interval bound from the uniform distribution.
            f0 = dmin + rand() * dmax
            # Sample additional interval bounds.
            f1 = f0 + sort(rand(1, ninter - 2), axis=1) * (pi - f0)
            finter[0, 1] = f0
            finter[0, 2:ninter] = f1[0, :]
            # Random shift of intervals.
            finter = rand() * pi + finter
            ind = nonzero(finter >= pi)
            if ind[0].size != 0:
                # Set interval bounds between 0 and pi by periodicity.
                k = ind[1][0]
                ind = concatenate((arange(k, ninter), arange(0, k)))
                finter[0, :] = finter[0, ind]
                k = ninter - k
                finter[0, 0:k] = finter[0, 0:k] - pi
                # Set the index of the step value corresponding to the
                # uniformly sampled interval.
                k = mod(k + 1, ninter)
            else:
                k = 1
            # Set interval bounds between -pi/2 and pi/2.
            finter = finter - pi / 2
            if self.steptrans:
                # Make sure that the interval sampled from uniform corresponds
                # to a step interval, and not to a transition.
                self.trans = mod(k + 1, 2)
                k = floor_divide(k, 2)

        self.finter[:, :] = finter[:, :]

        return k

    def SampleStepConstants(self, k=-1):
        """
        Sample constants within (0, 1).
        The simulation of the step constants depends on smode.
        - smode[0]='unif': step values are sampled from a uniform
        distribution on (smode[1], smode[2]).
        - smode[0]='unifmin': the minimal step value is sampled from a
        uniform distribution on (smode[1], smode[2]).
        - smode[0]='unifrange': the step value range is sampled from a uniform
        distribution on (0, smode[2] - smode[1]).

        Parameters
        ----------
        k : int, optional
            n index where to put the minimal value. The default is -1.

        Returns
        -------
        self.fparam.

        """
        if "step" not in self.ftype:
            print("pefunction.SampleStepConstants: only for step functions.")
            return(0)

        mode = self.smode[0]
        a = self.smode[1]
        b = self.smode[2]
        d = b - a
        constants = a + d * rand(1, self.fparam.size)

        if mode != 'unif' and self.fparam.size > 1:
            fmin = amin(constants, axis=None)
            fmax = amax(constants, axis=None)
            if ('min' in mode) or ('max' in mode):
                self.smode[0] = 'unifmin'
                u = a + rand(1) * d  # Minimal value, uniform in (a, b).
                v = rand(1) * (b - u)  # Value range in (0, b - u).
            else:
                self.smode[0] = 'unifrange'
                v = rand(1) * d  # Value range, uniform in (0, d).
                u = a + rand(1) * (d - v)  # Minimal value.

            constants = u + (constants - fmin) / (fmax - fmin) * v

            if 'max' in mode:
                self.smode[0] = 'unifmax'
                constants = 1 - constants

        if k != -1:
            val = constants[0, k]
            kmin = argmin(constants[0, :])
            constants[0, k] = constants[0, kmin]
            constants[0, kmin] = val

        self.fparam[:, :] = constants[:, :]

    def SetStepSampleMode(self,
                          mode_cst='unif', a=0, b=1,
                          mode_int='unif', d=0):
        """
        Set the simulation mode for sampling parameters of a step function.

        Parameters
        ----------
        mode_cst : str, optional
            The mode of simulation of constant values.
            - if mode_cst = 'unif':
                step values are sampled from a uniform distribution on (a, b).
            - if mode_cst = 'unifmin':
                the minimal step value is sampled from a uniform distribution
                on (a, b).
            - if mode_cst = 'unifrange':
                the step value range is sampled from a uniform distribution
                on (a, b).
            The default is 'unif'.

        a : float, optional
            lower bound. The default is 0.
        b : float, optional
            upperbound. The default is 1.
        mode_int : str, optional
            The mode of simulation of step bounds.
            - if mode_int='unif': the bounds are uniformly sampled
            over the interval (-pi/2, pi/2).
            - if mode_int='nonunif': the bounds are sampled so that the size
            of one of them is uniformly sampled.
            The default is 'unif'.
        d : float, optional
            Minimal value of interval size. The default is 0.

        Returns
        -------
        self.smode
        """
        if "step" not in self.ftype:
            print("pefunction.SetStepSampleMode: only for step functions.")
            return(0)

        self.smode = []
        self.smode.append(mode_cst)
        self.smode.append(a)
        self.smode.append(b)
        self.smode.append(mode_int)
        self.smode.append(d)

    def ComputeStepBasis(self):
        """
        Compute basis functions of a representation by a step function at
        position of self.t.

        Returns
        -------
        self.basis:
            a matrix where values of the kth basis function are on the kth row.
        """
        self.basis = zeros((self.fparam.size, self.values.size))
        for j in range(self.values.size):
            t = self.t[0, j]
            # Replace t in the interval (-pi/2, pi/2).
            if (t < - pi / 2) or (t >= pi / 2):
                t = t - floor((t + pi / 2) / pi) * pi
            # Find the interval which t belongs to.
            ind = nonzero(self.finter > t)
            if ind[1].size == 0:
                ind = self.finter.size
                ind0 = 0
            else:
                if self.steptrans:
                    ind0 = floor_divide(ind[1][0], 2)
                    ind = ind[1][0]
                else:
                    ind0 = ind[1][0]

            # Define a basis function value within a transition interval.
            if self.steptrans and (mod(ind, 2) == self.trans):
                if ind == self.finter.size:
                    f0 = self.finter[0, ind - 1]
                    f1 = self.finter[0, 0] + pi
                else:
                    if ind == 0:
                        f0 = self.finter[0, -1] - pi
                        f1 = self.finter[0, ind]
                    else:
                        f0 = self.finter[0, ind - 1]
                        f1 = self.finter[0, ind]
                t = (t - f0)
                t = t / (f1 - f0) * pi
                t = (1 - cos(t)) / 2

                ind0 = ind0 - 1 + self.trans
                ind0 = mod(ind0, self.fparam.size)
                ind1 = mod(ind0 + 1, self.fparam.size)
                self.basis[ind0, j] = 1 - t
                self.basis[ind1, j] = t
            else:
                self.basis[ind0, j] = 1

# Functions for the management of the Fourier representation.

    def InitFourierFunction(self, mode='init', M=3):
        """
        Define and update the Fourier representation of a periodic function.

        Parameters
        ----------
        mode : str, optional
            The utilisation mode ('init', 'update').
            Use the 'init' mode to set the function representation
            at random (default), or 'update' to change its parameters at
            random. The default is 'init'.
        M : int, optional
            M * 2 + 1 is the number of Fourier coefficients. The default is 3.

        Returns
        -------
        self.fparam.
        """

        if 'init' in mode:
            """
            Set the representation.
            """
            self.ftype = 'Fourier'
            if M <= 0:
                M = 3
            else:
                M = int(ceil(M))
            self.M = M
            self.fparam = zeros((1, 2 * M + 1))

        self.SampleFourierCoefficients()

    def SampleFourierCoefficients(self):
        """
        Sample the Fourier coefficients.

        Returns
        -------
        self.fparam.
        """
        if 'Fourier' not in self.ftype:
            print("perfunction.SampleFourier...: only for Fourier function.")
            return(0)
        M = floor_divide(self.fparam.size - 1, 2)
        param = arange(1, M + 1).reshape(1, M)
        param = 1 / sqrt(1 + power(param, 2))
        param = concatenate((param, param), axis=1)
        param = randn(1, 2 * M) * param
        param0 = sum(absolute(param), axis=None)
        param = concatenate((asmatrix(param0), param), axis=1)
        self.fparam[:, :] = param[:, :]

    def ComputeFourierBasis(self):
        """
        Compute basis functions of the Fourier representation.

        Returns
        -------
            self.basis:
            a matrix where values of the kth basis function are on the kth row.
        """
        M = self.M
        th = arange(1, M+1).reshape((M, 1))
        th = 2 * matmul(th, self.t.reshape((1, self.t.size)))
        th = concatenate((ones((1, th.shape[1])), cos(th), sin(th)), axis=0)
        self.basis = array(th)


def DiscreteFunctionDescription(values, delta):
    """
    Compute some features to describe a function represented in a dicrete way
    by successive values in a vector.

    Parameters
    ----------
    values : numpy.array
        Successive values of the function in a vector.
        delta: lag between values.
    delta : numpy.array

    Returns
    -------
    stats : ndarray
        Basics statistics: min, max, mean, median of the function.
    deviation : ndarray
        Measures of deviations: standard deviation,
        absolute deviations to the mean and the median.
    sharpness : ndarray
        Measures of sharpness computed from discrete derivative:
        discrete tv-norm, maximum of absolute discrete derivative.
    """
    fmin = amin(values)
    fmax = amax(values)
    fmean = mean(values)
    fmedian = median(values)
    stats = array([fmin, fmax, fmean, fmedian])
    # Measures of deviations.
    dev_std = std(values)
    dev_abs_mean = mean(absolute(values - fmean))
    dev_abs_median = mean(absolute(values - fmedian))
    deviation = array([dev_std, dev_abs_mean, dev_abs_median])
    # Discrete derivative of the function.
    dvalues = diff(values) / delta
    # TV norm of the function.
    dvalues = absolute(dvalues)
    tvnorm = mean(dvalues)
    # Maximum of the absolute derivatives.
    maxabs = amax(dvalues)
    sharpness = array([tvnorm, maxabs])

    return stats, deviation, sharpness
