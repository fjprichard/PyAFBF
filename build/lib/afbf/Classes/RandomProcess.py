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
"""Module for the management of random processes.

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>

.. note::
    This version only deals with the fractional Brownian motion.
    
"""

from afbf.utilities import pi, randn, rand, absolute, concatenate, fft, arange
from afbf.utilities import amin, reshape, mean
from afbf.utilities import real, cumsum, power, sqrt, plt, log, nonzero, zeros
from afbf import perfunction


class process:
    r"""
    This class handles random processes, with a focus on fractional Brownian
    motions (FBM).

    .. _process:

    A FBM is a Gaussian :term:`random process<random field>`
    with :term:`stationary<stationarity>` :term:`increments`. Its properties
    (:term:`regularity`, order of self-similarity, ...)
    are determined by a single parameter :math:`H \in (0, 1)`,
    called the Hurst index.


    :example:
        Simulation of a FBF with Hurst index :math:`H=0.2` at
        times :math:`\{1, \cdots, T\}`.

    .. code-block:: python

        from afbf import process
        model = process('fbm', param=0.2)
        model.Simulate(T=1000)
        model.Display(1)

    .. image:: ./Figures/mbf.png

    :param :ref:`ndarray` autocov: The autocovariance of the process.

    :param :ref:`ndarray` spect: The Fourier spectrum of the autocovariance.

    :param :ref:`ndarray` vario: The semi-variogram of the process.

    :param :ref:`ndarray` y: The values of a simulation of the process.
    """

    def __init__(self, type='fbm', param=-1):
        """Constructor method.

        :param type:  Type of processes ("fbm"). The default is 'fbm',
            the only process implemented in this package version.
        :type type: str, optional

        :param param:  Parameters of the process. The default is -1,
            meaning that it is uniformly sampled.
        :type param: float or :ref:`ndarray`, optional
        """
        self.type = type
        self.extended = False
        if param <= 0 or param >= 1:
            self.param = rand()
        else:
            self.param = param
        self.spect = None
        self.autocov = None
        self.vario = None
        self.y = None

    def ComputeAutocovariance(self, T=10):
        r"""Compute the autocovariance of the process increments
            at uniformly-spaced lags :math:`\{0, 1,  \cdots, T\}`.

        :param T: The maximal lag. Default is 10.
        :type T: int

        :returns: Attribute autocov.
        """
        if self.type == 'fbm':
            self.ComputeFBMAutocovariance(T)

    def ComputeAutocovarianceSpectrum(self):
        """Compute the Fourier spectrum of the periodized autocovariance.

        :returns: Attribute spect.
        """
        if self.autocov is None:
            self.ComputeAutocovariance()
            print('Warning: implicit run of ComputeAutocovariance.')

        # Periodization of the autocovariance.
        self.spect = concatenate((self.autocov,
                                  self.autocov[self.autocov.size-2:0:-1, :]),
                                 axis=0)
        # Computation of the spectrum.
        self.spect = fft.fft(self.spect, axis=0, norm=None)
        if (amin(real(self.spect)) <= 0):
            print('ComputeAutocovarianceSpectrum: negative eigenvalue.')

    def ComputeSemiVariogram(self, lags, logvario=0):
        """Compute the semi-variogram of the process at lags given in lags.

        :param :ref:`ndarray` lags: Lags where to compute the variogram.

        :param logvario:
            if logvario>0, a log semi-variogram is computed.
            The default is 0.

        :type logvario: int, optional

        :returns: Attribute vario.

        """
        if self.type == 'fbm':
            self.ComputeFBMSemiVariogram(lags, logvario)

    def ComputeFBMAutocovariance(self, T):
        r"""Compute the autocovariance of the increments of a fractional
        Brownian motion of Hurst index :math:`H` at
        uniformly-spaced lags :math:`\{0, 1, \cdots, T-1 \}`.

        :param T: The maximal lag.
        :type T: int

        :returns: Attribute autocov.
        """
        H = self.param
        H2 = 2 * H
        k = arange(0, T, 1).reshape(T, 1)
        self.autocov = 0.5 * (power(absolute(k + 1), H2) +
                              power(absolute(k - 1), H2) -
                              2 * power(absolute(k), H2))

    def ComputeFBMSemiVariogram(self, lags, logvario=0):
        """Compute the semi-variogram of the fbm at lags given in lags.

        :param :ref:`ndarray` lags: Lags where to compute the variogram.
        :param logvario: if logvario>0, a log semi-variogram is computed.
            The default is 0.
        :type logvario: int, optional

        :returns: Attribute vario.
        """
        if logvario == 0:
            self.vario = 0.5 * power(absolute(lags), 2 * self.param)
        else:
            v = power(absolute(lags), 2)
            ind = nonzero(v != 0)
            self.vario = zeros(lags.shape)
            self.vario[ind] = 0.5 * power(v[ind], self.param)
            v[ind] = log(v[ind])
            if logvario != 1:
                v[ind] = power(v[ind], logvario)
            self.vario[ind] = self.vario[ind] * v[ind]

    def ExtendFBM(self, M=5):
        r"""Extend the definition of a FBM to a non self-similar process.

        :param M: The number of parameters used for defining the extension
            function. The default is 5.
        :type M: int, optional

        :example: Simulation of an extended FBF with Hurst index :math:`H=0.2`
            at times :math:`\{1, \cdots, T\}`.

        .. code-block:: python

            from afbf import process
            model = process('fbm', param=0.2)
            model.ExtendFBM()
            model.Simulate(T=1000)
            model.Display(1)

        .. image:: ./Figures/embf.png

        """
        self.extended = True
        self.gf = perfunction('step-smooth', M)
        self.gf.SetUniformStepInterval()
        self.gn = 100000

    def Simulate(self, T):
        r"""Simulate the process at uniformly-spaced positions
            :math:`\{0, 1, \cdots, T\}`.

        :param T: The maximal lag.
        :type T: int

        :returns: Attribute y.
        """
        # Simulation of the process increments.
        self.Simulate_CirculantCovarianceMethod(T)
        # Simulation of the process by integration of increments.
        self.IntegrateProcess(1)

    def Simulate_CirculantCovarianceMethod(self, T):
        r"""Simulate process :term:`increments` at positions
        :math:`\{0, 1, \cdots, T\}`.

        .. note::

            The method was developed by Wood and Chan.
            It is described in :cite:p:`Wood1994`.

        :param T: The maximal lag.
        :type T: int

        :returns: Attribute y.
        """
        # Computation of the autocovariance of the process.
        self.ComputeAutocovariance(T)
        # Computation of its spectrum.
        self.ComputeAutocovarianceSpectrum()
        # A realization of the process.
        self.y = sqrt(self.spect)
        T2 = self.spect.size
        if self.extended:
            c = arange(0, T)
            self.gf.Evaluate(c / self.gn * pi - pi / 2)
            c = concatenate((self.gf.values,
                             self.gf.values[:, T-2:0:-1]),
                            axis=1)
            c = reshape(c, self.y.shape)
        else:
            c = 1
        self.y = fft.ifft(self.y * (randn(T2, 1) + 1j * randn(T2, 1)) * c,
                          axis=0, norm="ortho")

        # Realization of increments of the fractional Brownian motion.
        self.y = real(self.y[0:T, 0])
        self.y = self.y / sqrt(mean(power(self.y, 2), axis=None))

    def IntegrateProcess(self, order):
        """Integrate the process at a given order.

        :param int order: The order of integration.

        :returns: Attribute y.
        """
        if self.y is None:
            print("IntegrateProcess: Simulate a process before integrating.")
            return(0)

        for i in arange(0, order, 1):
            self.y = cumsum(self.y)
        self.y = self.y.reshape(self.y.size, 1)

    def Display(self, nfig=1):
        """Display the realization of the process.

        :param nfig: Figure index. The default is 1.
        :type nfig: int, optional
        """
        if self.y is None:
            print("Display: Simulate a process before integrating.")
            return(0)

        plt.figure(nfig)
        plt.plot(self.y)
