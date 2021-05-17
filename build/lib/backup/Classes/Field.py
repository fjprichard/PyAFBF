#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for the definition of anisotropic franctional Brownian fields
and the computation of its main features.
"""
from afbf.utilities import pi, linspace, unique, concatenate, sort, power
from afbf.utilities import ceil, amin, amax, mean, asmatrix, reshape, nonzero
from afbf.utilities import sum, multiply
from afbf.utilities import BETA_H
from afbf.Classes.PeriodicFunction import DiscreteFunctionDescription
from afbf import sdata, coordinates, perfunction


class field:
    r"""
    This class deals with the Anisotropic Fractional Brownian Fields (AFBF).

    These fields are Gaussian with stationary increments and have
    a semi-variogram characterized by a spectral density of the form
    ..math:: $$f(w)=c(\arg(w)) |w|^{-2h(\arg(w))-2}, w in \mathbb{R}^2,$$
             where $c$ and $h$ are pi-periodic functions both depending on
             the direction $\arg(w)$ of the frequency $w$.
    Functions $c$ and $h$ are called topothesy and Hurst functions, resp. They
    both determine the field properties.

    This field family includes the fractional Brownian field with a density
    ..math:: $$f(w) = c \vert w \vert^{-2H-2}, w in \mathbb{R}^2,$$
    where c and H are both constants.

    The field family also includes the elementary fractional Brownian field,
    which is anisotropic. Its density is of the form
    ..math:: $f(w) = c \vert w \vert^{-2H-2}$, if $\phi-d<\arg(w)<\phi+d$,
             and 0 otherwise,
             for $-pi/2<=\phi<=pi/2$, $d \in (0,pi/2]$ $H in (0,1)$, and $c>0$.

    References:

    [1] F. Richard, Tests of isotropy for rough textures of trended images,
        Statistica Sinica, 26:1279-1304, 2016.
    [2] F. Richard, Some anisotropy indices for the characterization of
        Brownian textures and their application to breast images,
        Spatial Statistics, 18:147--162, 2016.
    [3] F. Richard, Anisotropy of Hölder Gaussian random field:
        characterization, estimation and application to image textures,
        Statistics and Computing, 28(6):1155 - 1168, 2018.
    """

    def __init__(self, fname='fbf', topo=None, hurst=None):
        """
        See SetModel.
        """
        self.SetModel(fname, topo, hurst)

    def SetModel(self, fname='fbf', topo=None, hurst=None):
        """
        Set the field model either in a predefined or a customized mode.

        In 'customized' mode, the field is defined with topothesy and Hurst
        functions given in arguments as objects of the class perfunction.

        In 'predefined mode' mode, the topothesy and Hurst functions
        are generated at random according to the type of field indicated
        in fname. Available predefined fields are

        - 'fbf': fractional Brownian field (default).
        - 'efbf': elementary fractional Brownian field.
        - 'afbf': afbf with constant topothesy and step Hurst function.
        - 'afbf-smooth': afbf with constant topothesy and smooth step
                         Hurst function.
        - 'afbf-topo': afbf with step topothesy and constant Hurst function.
        - 'afbf-topo-smooth': afbf with smooth step topothesy and constant
                              Hurst function.
        - 'afbf-topo-Fourier': afbf with Fourier topothesy and constant
                               Hurst function.

        Parameters
        ----------
        fname : str, optional
            Name of the field. The default is 'fbf'.
        topo, hurst : perfunction, optional
            Topothesy and hurst functions of the field given in
            customized mode. The default are None.

        Returns
        -------
        self.topo, self.hurst
        """
        self.topo = None
        self.Hurst = None

        if hurst is None or topo is None:  # Predefined field.
            # Fractional Brownian field.
            if fname == 'fbf':
                self.fname = 'Fractional Brownian field'
                self.order = 0
                self.topo = perfunction('step-constant')
                self.hurst = perfunction('step-constant')

            # Elementary fractional Brownian field.
            elif fname == 'efbf':
                self.fname = 'Elementary fractional Brownian field'
                self.order = 0
                self.topo = perfunction('step-ridge', 1, 'init')
                self.hurst = perfunction('step-constant')

            # Some other AFBF.
            elif 'afbf' in fname:
                self.fname = 'Anisotropic fractional Brownian field'
                self.order = 0
                f1 = perfunction('step-constant')

                if 'smooth' in fname:
                    f2 = perfunction('step-smooth')
                else:
                    if 'Fourier' in fname:
                        f2 = perfunction('Fourier')
                    else:
                        f2 = perfunction('step-piecewise')

                if 'topo' in fname:
                    self.topo = f2
                    self.hurst = f1
                else:
                    self.topo = f1
                    self.hurst = f2
            else:
                print('Field.SetModel(): Unknown predefined field.')
                return(0)
        else:  # Customized mode of definition.
            if isinstance(topo, perfunction) and\
                    isinstance(hurst, perfunction):
                self.fname = fname
                self.topo = topo
                self.hurst = hurst
                self.FindOrder()
            else:
                print('Field.SetModel(): set hurst and topo as perfunction.')
                return(0)

        self.order = 0
        self.extended = False
        self.vario = None
        self.hurst.fname = 'Hurst function'
        self.topo.fname = 'Topothesy function'
        return(1)

    def CheckValidity(self):
        """
        Check if the field is valid.

        Returns
        -------
            valid: boolean.
            True if the field is valid and False otherwise.
        """
        valid = isinstance(self.topo, perfunction)\
            and isinstance(self.hurst, perfunction)

        if valid is False:
            print("The field is not properly defined.")

        return(valid)

    def DisplayParameters(self, nfig=1):
        """
        Plot the graph of the topothesy and Hurst functions of the field.

        Parameters
        ----------
        nfig : int, optional
            Figure index. The default is 1.

        """
        if self.CheckValidity():
            self.topo.Display(nfig)
            self.hurst.Display(nfig+1)
        else:
            return(0)

        if isinstance(self.vario, sdata):
            self.vario.Display(nfig+2)

    def Save(self, filename):
        """
        Save the model in file.

        Parameters
        ----------
        filename : str
            File name without extension.
        """
        self.hurst.Save(filename + '-hurst')
        self.topo.Save(filename + '-topo')

    def ComputeSemiVariogram(self, lags):
        """
        Compute values of the semi-variogram of the field
        at points given by lags.

        Parameters
        ----------
        lags : coordinates
            Lags at which to compute the semi-variogram.

        Returns
        -------
        self.vario : sdata
            Variogram.

        This function might be very slow when Hurst or topothesy
        function is not a step function.
        """
        if self.CheckValidity() is False:
            return(0)
        if self.order != 0:
            print('Field: The semi-variogram is defined for field of order 0.')
            return(0)
        if not isinstance(lags, coordinates):
            print('Definition: the lags must be coordinates.')
            return(0)

        c = self.topo
        h = self.hurst

        if ('Fourier' in c.ftype) or ('smooth' in c.ftype) or\
                ('Fourier' in h.ftype) or ('smooth' in h.ftype):
            t = linspace(- pi / 2, pi / 2, 10000)
        else:
            # Determine the common step intervals of the topothesy and Hurst
            # functions.
            t = sort(concatenate((c.fparam[0, :], h.fparam[0, :]), axis=0),
                     axis=0)
            t[0] = - pi / 2
            t[-1] = pi / 2
            t = t[nonzero(t <= pi / 2)]
            t = t[nonzero(t >= -pi / 2)]
            t = unique(t)

        # Evaluate the topotothesy and Hurst functions at positions t.
        c.Evaluate(t)
        h.Evaluate(t)

        # Create a sdata object to store the semi-variogram.
        self.vario = sdata(lags)
        self.vario.name = 'Field semi-variogram.'

        coord2 = asmatrix(power(lags.xy[:, 0], 2) + power(lags.xy[:, 1], 2))
        coord2 = reshape(coord2, (coord2.size, 1))
        N = lags.N

        for k in range(c.t.size-1):
            H = h.values[0, k]
            C = c.values[0, k]
            if C != 0 and H != 0:
                self.vario.values = self.vario.values +\
                    C * pow(2, (2 * H - 1)) / pow(N, 2 * H) *\
                    multiply(BETA_H(lags, c.t[0, k], c.t[0, k+1], H),
                             power(coord2, H))

        return(1)

    def FindOrder(self):
        """
        Find the order of the intrinsic field.

        This function is only available when the Hurst function is a step
        function.
        """
        if self.CheckValidity() is False:
            return(0)

        if 'step' in self.hurst.ftype:
            inter = linspace(-pi/2, pi/2, 10000)
            self.hurst.Evaluate(inter)
            self.topo.Evaluate(inter)
            ind = nonzero(self.topo.values[:] != 0)
            self.order = ceil(amax(self.hurst.values[ind])) - 1
            return ind
        else:
            print('FindOrder(): only available for Hurst step function')
            return 0

    def ChangeOrder(self, neworder):
        """
        Change the order of the intrinsic field.

        Parameters
        ----------
            neworder: int
                The new order of the field.

        This function is only available when the Hurst function is a step
        function.
        """
        if self.CheckValidity() is False:
            return(0)

        if ('step' in self.hurst.ftype):
            order0 = self.order
            ind = nonzero(self.hurst.fparam[0, :] != -1)
            self.hurst.fparam[0, ind] = (neworder + 1) / (order0 + 1) *\
                self.hurst.fparam[0, ind]
            self.FindOrder()
            return(1)
        else:
            print('ChangeOrder(): only available for Hurst step function')
            return(0)

    def ComputeFeatures(self):
        """
        Compute several intrinsic features of the field.

        Returns
        -------
            To be completed.
        """
        if self.CheckValidity() is False:
            return(0)

        # Compute features of the Hurst and topothesy functions.
        self.hurst.ComputeFeatures()
        delta = self.topo.ComputeFeatures()
        stopo = sum(self.topo.values)
        m = self.topo.t.size

        # Anisotropy indices for topothesy and Hurst functions
        self.aniso_indices_topo = self.topo.dev
        self.aniso_sharpness_topo = self.topo.sharpness
        self.aniso_indices_hurst = self.hurst.dev * 2
        self.aniso_sharpness_hurst = self.hurst.sharpness

        # Mixed anisotropy indices.
        tvalues = self.topo.values / stopo  # Topothesy normalization.
        hvalues = 2 * self.hurst.values - 2
        density1 = tvalues * power(0.5, hvalues)
        density2 = tvalues * power(1.5, hvalues)
        [s1, d1, sh1] = DiscreteFunctionDescription(density1, delta)
        [s2, d2, sh2] = DiscreteFunctionDescription(density2, delta)
        self.aniso_indices_mixed_1 = d1
        self.aniso_indices_mixed_2 = d2
        self.aniso_sharpness_mixed_1 = sh1
        self.aniso_sharpness_mixed_2 = sh2

        # Analysis of the Hurst function.
        # Find the support of the topothesy function on (-pi/2, pi/2).
        ind = nonzero(self.topo.values != 0)
        self.topo.values = self.topo.values[ind]
        self.hurst.values = self.hurst.values[ind]
        self.hurst.t = self.hurst.t[ind]

        # Minimum and maximum of the Hurst function on this support.
        self.H = amin(self.hurst.values)
        self.Hmax = amax(self.hurst.values)

        # Argmin set of the Hurst function.
        ind = nonzero(self.hurst.values == self.H)

        # Length of the argmin set.
        self.hurst_argmin_len = ind[0].size / m

        # An anisotropy index based on this length.
        self.hurst_aniso_index = (0.5 - abs(0.5 - self.hurst_argmin_len)) * 2

        # Center of the argmin set.
        ind2 = nonzero(self.hurst.values != self.H)
        if ind2[0].size != 0:
            t0 = ind2[0][-1]
        else:
            t0 = self.hurst.values.size
        self.hurst.t[0:t0] = self.hurst.t[0:t0] + pi
        hmean = mean(self.hurst.t[ind])
        if hmean < - pi / 2:
            hmean = hmean + pi
        elif hmean >= pi / 2:
            hmean = hmean - pi
        self.hurst_argmin_mean = hmean
