#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
# Credits
# #######
#
# Copyright(c) 2021-2021
# ----------------------
#
# * Institut de Mathématiques de Marseille <https://www.i2m.univ-amu.fr/>
# * Université d'Aix-Marseille <http://www.univ-amu.fr/>
# * Centre National de la Recherche Scientifique <http://www.cnrs.fr/>
#
# Contributors
# ------------
#
# * `Frédéric Richard <mailto:frederic.richard@univ-amu.fr>`_
#
#
# * This module is part of the package PyAFBF.
#
# Licence
# -------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ######### COPYRIGHT #########
r"""Module for the management of non-negative :math:`\pi`-periodic functions.

.. codeauthor:: Frédéric Richard <frederic.richard@univ-amu.fr>
"""
from afbf.utilities import linspace, pi, concatenate, reshape, nonzero
from afbf.utilities import asmatrix, cos, sin
from afbf.utilities import rand, plt, zeros, amin, amax, argmin, sort, floor
from afbf.utilities import ceil, sqrt, power, randn
from afbf.utilities import sum, diff, mod, floor_divide, mean, std, median
from afbf.utilities import absolute, ones, matmul, arange, array, ndarray


class perfunction:
    r"""This class handles :math:`\pi`-periodic positive parametric functions.

    .. _perfunction:

    The available function representations are

    - the representation in a Fourier basis,
    - a representation by step functions,
    - a representation by smoothed step functions.

    In the Fourier representation, the function is defined as

    .. math:: f(t) = a_0 + \sum_{m=1}^M a_{m, 1} \cos(2mt)+a_{m, 2} \sin(2mt),

    where :math:`a_0` and :math:`a_{m, k}` are real parameters.

    A step function is defined as

    .. math::
        f(t) =
        a_0 \left(
            \delta_{[-\frac{\pi}{2}, \psi_1)}(t)
            + \delta_{[\psi_{M+1}, \frac{\pi}{2})}(t)
            \right)
        + \sum_{m=1}^M a_i \delta_{[\psi_i, \psi_{i+1})}(t),

    where :math:`a_i` are non-negative parameters, and :math:`\psi_i` are
    ordered angles in :math:`[-\frac{\pi}{2}, \frac{\pi}{2})`.

    The definition of step functions can be adapted to include smooth
    transitions between intervals where the function is constant.

    .. note::
        Representations are defined by an expansion of the form

        .. math:: f(t) = \sum_{m=0}^M \alpha_m B_m(t)

        for some coefficients :math:`\alpha_m` and basis functions
        :math:`B_m`.

    :example: Definition of a function using the Fourier representation
        with :math:`M=3` (i.e. 7 coefficients).

    .. code-block:: python

        from afbf import perfunction
        f1 = perfunction('Fourier', 3, 'Fourier')
        f1.Display(1)

    .. image:: ./Figures/function-fourier.png

    :example: Definition of a function with three steps.

    .. code-block:: python

        from afbf import perfunction
        f2 = perfunction('step', 3, 'step function')
        f2.Display(2)

    .. image:: ./Figures/function-step.png

    :example: Definition of a smooth function with two steps.

    .. code-block:: python

        from afbf import perfunction
        f3 = perfunction('step-smooth', 2, 'smooth step function')
        f3.Display(3)

    .. image:: ./Figures/function-smooth-step.png

    :example: Definition of a step function with a ridge.

    .. code-block:: python

        from afbf import perfunction
        f4 = perfunction('ridge-step', 1, 'ridge function')
        f4.Display(4)

    .. image:: ./Figures/function-ridge.png

    :param str fname: Name of the function.

    :param str ftype: Type of the function representation.
        Some predefined type are available:

            - 'step': step function.
            - 'step-constant': constant function.
            - 'step-ridge': a step function with ridges.
            - 'step-smooth': smooth step function.
            - 'Fourier': Fourier representation.

    :param fparam: Representation parameters :math:`\alpha_m`.
    :type fparam: :ref:`ndarray`

    :param finter: Interval bounds :math:`\psi_i` for a step function.
    :type finter: :ref:`ndarray`

    :param boolean steptrans: True if there are transitions between step.

    :param int trans:
        indicate where step transitions are on even or odd intervals ({0, 1}).

    :param basis: an array where values of the mth basis function :math:`B_m`
        of the representation are on the mth row.
    :type basis: :ref:`ndarray`

    :param t: positions at which to evaluate the function.
    :type t: :ref:`ndarray`

    :param stats: Basics statistics;
        min, max, mean, median of the function.
    :type stats: :ref:`ndarray`

    :param dev: Measures of deviations of the function; standard deviation,
            absolute deviations to the mean and to the median.
    :type dev: :ref:`ndarray`

    :param sharpness: Measures of sharpness computed on discrete function
        derivative (discrete tv-norm, maximum of absolute discrete derivative).
    :type sharpness: :ref:`ndarray`

    :param smode: Simulation mode of a step function (see SetStepSampleMode_).

    :param float translate:
        Translation to be applied to the function (defaut to 0).

    :param float rescale:
        Factor of a rescaling to be applied to the function.
    """

    def __init__(self, ftype='step-smooth', param=0, fname='noname'):
        """Constructor method.

        Define a positive periodic function with a specific representation.

        :param str ftype: A predefined function representation among
            - 'step',
            - 'step-constant',
            - 'step-ridge',
            - 'step-smooth',
            - and 'Fourier'.

            The default is 'step-smooth'.

        :param param: The number of parameters. Default to 0.
        :type param: int (, optional)

        :param fname: The name of the function. Default 'noname'.
        :type fname: str (, optional)
        """
        if not isinstance(fname, str) or not isinstance(ftype, str):
            print('perfunction: provide string arguments for fname and ftype')

        self.fname = fname
        self.ftype = 'undefined'
        self.fparam = None
        self.finter = None
        self.translate = 0
        self.rescale = 1

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
        """Check the validity of the periodic function.

        :returns: True if attributes are properly defined.
        :rtype: boolean
        """
        valid = isinstance(self.fparam, ndarray)
        if "step" in self.ftype:
            valid = valid and isinstance(self.finter, ndarray)

        if valid is False:
            print("The periodic function is not properly defined.")

        return(valid)

    def ShowParameters(self):
        """Show the parameters of the function.
        """
        if self.fparam is not None:
            print('Coefficient values:')
            print(self.fparam)

            if 'step' in self.ftype:
                print('Step interval bounds:')
                print(self.finter)

    def ChangeParameters(self, fparam=None, finter=None):
        """Change parameters of the function while keeping its
        representation.

        If parameters are not given, the parameters are changed at random.

        :param fparam: Parameters of the function. The default is None.
        :type fparam: :ref:`ndarray`, optional.

        :param finter: Step interval bounds (only for step functions).
            The default is None.
        :type finter: :ref:`ndarray`, optional

        :returns: Attributes fparam.
        """

        if fparam is None and finter is None:  # Change parameters at random.
            if self.ftype == 'Fourier':
                self.InitFourierFunction('update')
            if 'step' in self.ftype:
                self.InitStepFunction(self.ftype, 'update')
        else:  # Change parameters manually.
            if fparam is not None:
                if isinstance(fparam, ndarray):
                    if fparam.size == self.fparam.size:
                        self.fparam = reshape(fparam, self.fparam.shape)
                    else:
                        print('ChangeParameters: fparam size is not correct.')
                        return(0)
                else:
                    print('ChangeParameters: fparam should be ndarray.')
                    return(0)

            if 'step' in self.ftype and finter is not None:
                if isinstance(finter, ndarray):
                    if finter.size == self.finter.size:
                        self.finter = reshape(finter, self.finter.shape)
                    else:
                        print('ChangeParameters: finter size is not correct.')
                        return(0)
                else:
                    print('ChangeParameters: finter should be ndarray.')
                    return(0)

    def ApplyTransforms(self, translate=None, rescale=None):
        """Apply translation and/or rescaling transform to the function.

        :param float translate: Translation.
        :param float rescale: scaling factor (must be positive).

        :returns: Attributes translate, rescale.
        """
        if translate is not None and isinstance(translate, float):
            self.translate = translate

        if rescale is not None and isinstance(rescale, float):
            if rescale > 0:
                self.rescale = rescale
            else:
                print('Rescaling factor must be positive.')

    def Evaluate(self, t=None):
        """Evaluate the function at some positions.

        :param t:  Positions at which the function is computed.
        :type t: :ref:`ndarray`, optional.

        :returns: Attribute values.

        .. note::

          If parameter t is omitted, the function is evaluated
          at points of the previous call of the function.
        """
        if self.CheckValidity() is False:
            return(0)

        if t is not None:
            if isinstance(t, ndarray) is False:
                print('pefunction.Evaluate: set parameter t as ndarray.')
                return(0)
            self.t = reshape(t, (1, t.size))

            # Take into account possible translation and rescaling.
            if self.translate != 0:
                self.t = self.t - self.translate
            if self.rescale != 1:
                self.t = self.t * self.rescale

            self.values = zeros((1, self.t.size))
            if 'step' in self.ftype:
                self.ComputeStepBasis()
            if 'Fourier' in self.ftype:
                self.ComputeFourierBasis()

            if self.rescale != 1:
                self.t = self.t / self.rescale
            if self.translate != 0:
                self.t = self.t + self.translate

        if self.basis is None:
            print("PeriodicFunctions.Evaluate: give positions t as ndarray.")
        else:
            self.values = matmul(self.fparam, self.basis)

    def Display(self, nfig=1):
        """Plot the graph of the function.

        :param nfig: The index of the figure. Default to 1.
        :type nfig: int, optional
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
        """Compute some features describing the function.

        :param m:   Number of discrete positions to evaluate the function.
            Default to 10000.
        :type m: int, optional

        :returns: Precision of the evaluation.
        :rtypes: scalar

        :returns: Attributes stats, dev, sharpness.
        """
        if self.CheckValidity() is False:
            return(0)

        t = linspace(- pi / 2, pi / 2, m)
        self.Evaluate(t)
        delta = pi / m
        stats, dev, sharpness =\
            DiscreteFunctionDescription(self.values, delta)
        self.stats = stats
        self.dev = dev
        self.sharpness = sharpness

        return delta

# Functions for the management of the representation by step functions.

    def InitStepFunction(self, ftype='step-constant', mode='init', M=2):
        """Define or update a step function at random.

        :param ftype: The type of step function.
            ('step','step-constant','step-ridge','step-smooth').
            The default is 'step-constant'.
        :type ftype: str, optional

        :param mode: The utilisation mode ('init', 'update').
            The default is 'init'.
        :type mode: str, optional

        :param M: Number of steps. The default is 2.
        :type M: int, optional

        :returns: Attributes fparam, finter.
        """
        if 'init' in mode:
            """
            Set the representation.
            """
            self.ftype = 'step'
            # Set the number of intervals.
            if ('constant' in ftype):
                M = 1
                self.ftype = 'step-constant'
            else:
                if M <= 0:
                    M = 2
                else:
                    M = int(ceil(M))

            if ('ridge' in ftype):
                M = 2 * M
                M2 = M
                self.ftype = 'step-ridge'
                self.steptrans = False
            elif ('smooth' in ftype):
                M2 = 2 * M
                self.steptrans = True
                self.ftype = 'step-smooth'
            else:
                M2 = M
                self.steptrans = False

            self.fparam = zeros((1, M))
            self.finter = zeros((1, M2))

            # Set transition intervals for smooth and ridge step functions.
            if ('ridge' or 'smooth' in self.ftype):
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
        """Sample interval bounds of a step function.

        :returns:  The index of the interval whose size is uniformly sampled.
        :rtype: int

        .. note::

            The simulation of the step constants depends
            on the attribute smode:

                - smode[3]='unif':
                    the bounds are uniformly sampled over
                    the interval (-pi/2, pi/2).
                - smode[3]='nonunif':
                    the bounds are sampled so that the
                    size of one of them is uniformly sampled.

            smode[4] is a minimal interval size.

            The mode of simulation can be changed using
            SetStepSampleMode_.
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
        """Sample constants within (0, 1).

        :param k: index where to put the minimal value. The default is -1.
        :type k: int, optional

        :returns: Attribute fparam.

        .. note::

             The simulation of the step constants depends on
             the attribute smode:

                 - smode[0]='unif':
                     step values are sampled from a uniform
                     distribution on (smode[1], smode[2]).
                 - smode[0]='unifmin':
                     the minimal step value is sampled from a
                     uniform distribution on (smode[1], smode[2]).
                 - smode[0]='unifrange':
                     the step value range is sampled from a
                     uniform distribution on (0, smode[2] - smode[1]).

             The mode of simulation can be changed using
             SetStepSampleMode_.
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
        r"""Set the simulation mode for sampling parameters of a step function.

        .. _SetStepSampleMode:

        :param mode_cst: The mode of simulation of constants :math:`a_m`:

            - if mode_cst = 'unif':
                step values are sampled from a uniform distribution on (a, b).
            - if mode_cst = 'unifmin':
                the minimal step value is sampled from a uniform distribution
                on (a, b).
            - if mode_cst = 'unifrange':
                the step value range is sampled from a uniform distribution
                on (a, b).

            The default is 'unif'.

        :type mode_cst: str, optional

        :param a: lower bound. The default is 0.
        :type a: float, optional

        :param b: upper bound. The default is 1.
        :type b: float, optional

        :param mode_int:   The mode of simulation of interval bounds
            :math:`\psi_m`:

            - if mode_int='unif':
                the bounds are uniformly sampled over the interval
                (-pi/2, pi/2).
            - if mode_int='nonunif':
                the bounds are sampled so that the size
                of one of them is uniformly sampled.

            The default is 'unif'.

        :type mode_int: str, optional

        :param d: Minimal value of interval size. The default is 0.
        :type d: float, optional

        :returns: Attribute smode.
        """
        if "step" not in self.ftype:
            print("perfunction.SetStepSampleMode: only for step functions.")
            return(0)

        self.smode = []
        self.smode.append(mode_cst)
        self.smode.append(a)
        self.smode.append(b)
        self.smode.append(mode_int)
        self.smode.append(d)

    def SetUniformStepInterval(self):
        """Set uniform step intervals.

        :returns: Attribute finter, trans.
        """
        if "step" not in self.ftype:
            print("SetUniformStepInverval: only for step functions.")
            return(0)

        self.ChangeParameters(
            None,
            linspace(-pi / 2, pi / 2, self.finter.size + 2)[1:-1]
            )
        if self.steptrans:
            self.trans = 1

    def ComputeStepBasis(self):
        """Compute basis functions of a representation by a step function at
        positions given in attribute t.

        :returns: Attribute basis.
        """
        self.basis = zeros((self.fparam.size, self.values.size))
        for j in range(self.values.size):
            t = self.t[0, j]
            # Set t in the interval (-pi/2, pi/2).
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
        """Define or update the Fourier representation of a periodic function.

        :param mode:  The utilisation mode ('init', 'update').
            Use the 'init' mode to set the function representation
            at random (default), or 'update' to change its parameters at
            random. The default is 'init'.
        :type mode: str, optional

        :param M: M * 2 + 1 is the number of Fourier coefficients.
            The default is 3.
        :type M: int, optional

        :returns: Attribute fparam.
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
        """Sample the Fourier coefficients.

        :returns: Attribute fparam.
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
        """Compute basis functions of the Fourier representation.

        :returns: Attribute basis.
        """
        M = self.M
        th = arange(1, M+1).reshape((M, 1))
        th = 2 * matmul(th, self.t.reshape((1, self.t.size)))
        th = concatenate((ones((1, th.shape[1])), cos(th), sin(th)), axis=0)
        self.basis = array(th)


def DiscreteFunctionDescription(values, delta):
    """Compute some features to describe a function represented
    in a dicrete way by successive values in a vector.

    :param values:  Successive values of the function in a vector.
    :type values: :ref:`ndarray`

    :param delta: lag between values.
    :type delta: :ref:`ndarray`

    :returns: stats, deviation and sharpness:

         - basics statistics of the function (min, max, mean, median).
         - measures of deviations (standard deviation, absolute deviations
           to the mean and the median).
         - measures of sharpness computed from discrete derivative
           (discrete tv-norm, maximum of absolute discrete derivative).

    :rtype: list of :ref:`ndarray`
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
