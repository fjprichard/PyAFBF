#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for the simulation and estimation of random processes, including the
fractional Brownian motion.
"""

from afbf.utilities import pi, randn, rand, absolute, concatenate, fft, arange
from afbf.utilities import amin
from afbf.utilities import real, cumsum, power, sqrt, plt, log, nonzero, zeros
from afbf import perfunction


class process:
    r"""
    Handle random processes, with a focus on fractional Brownian
    motions (FBM). It includes a method to simulate such processes.

    The FBM are Gaussian random process whose increments are stationary and
    have an autocovariance of the form
    ..math:: $$ c(t) = c_0 \vert t \vert^{H}, $$
             where $c_0>0$ and $H \in (0,1)$ is the Hurst parameter.
    """

    def __init__(self, type='fbm', param=-1):
        """
        Parameters
        ----------
        type : str, optional
            Type of processes ("fbm"). The default is 'fbm', the only
            process implemented in this package version.
        param : float or numpy.array, optional
            Parameters of the process. The default is -1, menanng that it is
            uniformly sampled.

        Returns
        -------
        None.

        """
        self.type = type
        self.extended = False
        if param <= 0 or param >= 1:
            self.param = rand()
        else:
            self.param = param

    def ComputeAutocovariance(self, T):
        """
        Compute the autocovariance of the process at uniformly-spaced
        lags (0, 1, ..., T).

        Parameters
        ----------
        T : int.
            Thee maximal lag.

        Returns
        -------
        self.autocov
        """
        if self.type == 'fbm':
            self.ComputeFBMAutocovariance(T)

    def ComputeAutocovarianceSpectrum(self):
        """
        Compute the Fourier spectrum of the periodized autocovariance.

        Returns
        -------
        self.spect.
        """
        # Periodization of the autocovariance.
        self.spect = concatenate((self.autocov,
                                  self.autocov[self.autocov.size-2:0:-1, :]),
                                 axis=0)
        # Computation of the spectrum.
        self.spect = fft.fft(self.spect, axis=0, norm=None)
        if (amin(real(self.spect)) <= 0):
            print('ComputeAutocovarianceSpectrum: negative eigenvalue.')

    def ComputeSemiVariogram(self, lags, logvario=0):
        """
        Compute the semi-variogram of the process at lags given in lags.

        Parameters
        ----------
        lags : numpy.array
            Lags where to compute the variogram.
        logvario : int, optional
            if logvario>0, a log semi-variogram is computed. The default is 0.

        Returns
        -------
        self.vario.

        """
        if self.type == 'fbm':
            self.ComputeFBMSemiVariogram(lags, logvario)

    def ComputeFBMAutocovariance(self, T):
        """
        Compute the autocovariance of the increments of a fractional Brownian
        motion of Hurst index H at uniformly-spaced lags (0, 1,..., T-1).

        Parameters
        ----------
        T : int
            Maximal lag.

        """
        H = self.param
        H2 = 2 * H
        k = arange(0, T, 1).reshape(T, 1)
        self.autocov = 0.5 * (power(absolute(k + 1), H2) +
                              power(absolute(k - 1), H2) -
                              2 * power(absolute(k), H2))

    def ComputeFBMSemiVariogram(self, lags, logvario=0):
        """
        Compute the semi-variogram of the fbm at lags given in lags.

        Parameters
        ----------
        lags : TYPE
            Lags where to compute the variogram.
        logvario : TYPE, optional
            if logvario>0, a log semi-variogram is computed. The default is 0.

        Returns
        -------
        self.vario.

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
        """
        Extend the definition of a FBM to a non self-similar process.

        Parameters
        ----------
        M : int, optional
            The number of parameters used for defining the extension function.
            The default is 5.

        """
        self.extended = True
        self.gf = perfunction('Fourier', M)
        self.gn = 100000

    def Simulate_CirculantCovarianceMethod(self, T):
        """
        Simulate the process increments at uniformly-spaced positions
        (0, 1, ..., T).
        The method used is described in
        Chan, G. & Wood, A.
        Simulation of Stationary Gaussian Processes in [0,1]d
        Journal of Computational and Graphical Statistics
        Vol. 3, No. 4 (Dec., 1994).

        Parameters
        ----------
        T : int
            Maximal lag.

        Returns
        -------
        self.y: ndarray
            values of the simulated process.

        """
        # Computation of the autocovariance of the process.
        self.ComputeAutocovariance(T)
        # Computation of its spectrum.
        self.ComputeAutocovarianceSpectrum()
        # A realization of the process.
        self.y = sqrt(self.spect)
        T2 = self.spect.size
        if self.extended:
            c = arange(0, T2)
            self.gf.Evaluate(c / self.gn * pi)
            c = self.gf.values.reshape(self.y.shape)
        else:
            c = 1
        self.y = fft.ifft(self.y * (randn(T2, 1) + 1j * randn(T2, 1)) * c,
                          axis=0, norm="ortho")

        # Realization of a fractional Brownian motion.
        self.y = real(self.y[0:T, 0])

    def IntegrateProcess(self, order):
        """
        Integrate the process at a given order.

        Parameters
        ----------
        order : int
            The order of integration.

        Returns
        -------
        self.y

        """
        for i in arange(0, order, 1):
            self.y = cumsum(self.y)
        self.y = self.y.reshape(self.y.size, 1)

    def Display(self, nfig=1):
        """
        Display the realization of the process.

        Parameters
        ----------
        nfig : int, optional
            Figure index. The default is 1.
        """
        plt.figure(nfig)
        plt.plot(self.y)
