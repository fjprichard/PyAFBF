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

r"""Module for the management of anisotropic fractional Brownian fields.

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>

"""
from afbf.utilities import pi, linspace, unique, concatenate, power
from afbf.utilities import ceil, amin, amax, mean, reshape, nonzero
from afbf.utilities import sum, zeros, array
from afbf.Classes.PeriodicFunction import DiscreteFunctionDescription
from afbf import sdata, coordinates, perfunction


class field:
    r"""This class handles Anisotropic Fractional Brownian Fields (AFBF).

    An AFBF is a Gaussian :term:`random field` with
    :term:`stationary<stationarity>` :term:`increments`
    whose probabibility distribution is characterized by
    a :term:`density` of the form
    :cite:p:`Bonami2003,Bierme08-ESAIM,Richard-2017,Richard-2016,Richard-2015b,
    Richard-2015,Richard-2010`

    .. math::
        v(x) = \frac{1}{2} \int_{-\pi/2}^{\pi/2}
        \tau(\theta)
        \left\vert
        \langle x, (\cos(\theta), \sin(\theta)) \rangle
        \right\vert^{2\beta(\theta)} d\theta

    where :math:`\tau` and :math:`\beta` are non-negative :math:`\pi`-periodic
    functions depending both on the direction :math:`\arg(w)` of
    the frequency :math:`w`.

    Functions :math:`\tau` and :math:`\beta` are called the
    **topothesy function** and the **Hurst function**, respectively.

    :param str fname: Name of the field.

    :param topo: The topothesy function :math:`\tau` of the field.
    :type topo: :ref:`perfunction<perfunction>`

    :param hurst: The Hurst function :math:`\beta` of the field.
    :type hurst: :ref:`perfunction<perfunction>`

    :param int order: The order of the field (used for the extension to
                      :term:`intrinsic` field
                      :cite:p:`Chiles-2012,Richard-2017,Richard-2016,
                      Richard-2015b,Richard-2015,Richard-2010`).

    :param scalar H: The Hurst index of the field.

    :param scalar hurst_argmin_len:
        The lenght of the argmin set of the Hurst function.

    :param scalar Hmax: The maximum of the Hurst function.

    :param scalar hurst_aniso_index:
        An anisotropy index derived from the lenght of the argmin set of
        the Hurst function.

    :param aniso_indices_topo: Measures of deviation of the topothesy function.
    :type aniso_indices_topo: :ref:`ndarray

    :param aniso_indices_hurst: Measures of deviation of the Hurst function.
    :type aniso_indices_hurst: :ref:`ndarray`

    :param aniso_sharpness_topo: Measures of sharpness of the topothesy
        function.
    :type aniso_sharpness_topo: :ref:`ndarray`

    :param aniso_sharpness_hurst: Measures of sharpness of the Hurst function.
    :type aniso_sharpness_hurst: :ref:`ndarray`

    :param aniso_indices_mixed_1: Measures of deviations of a function
        combining the topothesy and Hurst functions.
    :type aniso_indices_mixed_1: :ref:`ndarray`

    :param aniso_indices_mixed_2: Measures of deviations of another function
        combining the topothesy and Hurst functions.
    :type aniso_indices_mixed_2: :ref:`ndarray`

     :param aniso_sharpness_mixed_1: Measures of sharpness of a function
        combining the topothesy and Hurst functions.
    :type aniso_sharpness_mixed_1: :ref:`ndarray`

    :param aniso_sharpness_mixed_2: Measures of sharpness of another function
        combining the topothesy and Hurst functions.
    :type aniso_sharpness_mixed_2: :ref:`ndarray`
    """

    def __init__(self, fname='fbf', topo=None, hurst=None):
        """Constructor method.

        Set the :term:`random field` model either in a predefined or a
        customized mode.

        In customized mode, the field is defined with topothesy and Hurst
        functions given in arguments as objects of the class perfunction.

        In predefined mode, the topothesy and Hurst functions
        are generated at random according to the type of field indicated
        in fname. Available predefined fields are:

        - 'fbf':
            standard fractional Brownian field (default).
        - 'efbf':
            elementary fractional Brownian field.
        - 'afbf':
            Normalized afbf with a step Hurst function.
        - 'afbf-smooth':
            Normalized afbf with a smooth step Hurst function.
        - 'afbf-Fourier':
            afbf with a Fourier topothesy and a step Hurst function.
        - 'afbf-smooth-Fourier':
            afbf with a Fourier topothesy and a smooth step Hurst function.

        :param fname:  Name of the field. The default is 'fbf'.
        :type fname: str, optional

        :param topo: Topothesy function. Default is None.
        :type topo: :ref:`perfunction<perfunction>`, optional

        :param hurst: Topothesy function. Default is None.
        :type hurst: :ref:`perfunction<perfunction>`, optional

        :returns: Attributes fname, order, topo, hurst.
        """
        self.SetModel(fname, topo, hurst)

    def SetModel(self, fname='fbf', topo=None, hurst=None):
        """See Constructor method.
        """
        self.topo = None
        self.hurst = None

        if hurst is None or topo is None:  # Predefined field.
            # Standard fractional Brownian field.
            if fname == 'fbf':
                self.fname = 'Fractional Brownian field'
                self.order = 0
                self.topo = perfunction('step-constant')
                self.hurst = perfunction('step-constant')
                self.NormalizeModel()

            # Elementary fractional Brownian field.
            elif fname == 'efbf':
                self.fname = 'Elementary fractional Brownian field'
                self.order = 0
                self.topo = perfunction('step-ridge', 1)
                self.hurst = perfunction('step-constant')

            # Some other AFBF.
            elif 'afbf' in fname:
                self.fname = 'Anisotropic fractional Brownian field'
                self.order = 0
                if 'smooth' in fname:
                    self.hurst = perfunction('step-smooth')
                else:
                    self.hurst = perfunction('step')
                if 'Fourier' in fname:
                    self.topo = perfunction('Fourier')
                else:
                    self.topo = perfunction(self.hurst.ftype)
                    self.NormalizeModel()
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
            self.FindOrder()

        self.extended = False
        self.vario = None
        self.hurst.fname = 'Hurst function'
        self.topo.fname = 'Topothesy function'
        return(1)

    def NormalizeModel(self):
        """Normalize the model.

        .. note::
            This function can only be applied if the Hurst function is a
            step function.
        """
        if 'step' in self.hurst.ftype:
            # Redefining the topothesy function.
            topo = perfunction(self.hurst.ftype, self.hurst.fparam.size)
            self.topo = topo
            topo.fname = 'Topothesy function'
            topo.finter = self.hurst.finter
            topo.fparam = zeros(self.hurst.fparam.shape)
            if self.hurst.steptrans:
                topo.trans = self.hurst.trans
                topo.steptrans = self.hurst.steptrans

            coord = coordinates()
            coord.DefineNonUniformLocations(array([[1, 1]]))
            for j in range(self.hurst.fparam.size):
                h = self.hurst.fparam[0, j]
                if h > 0:
                    c = pow(2, - (2 * h)) / BETA_H(coord, -pi / 2, pi / 2, h)
                    topo.fparam[0, j] = c[0, 0]
        else:
            print('Warning: normalize only with step Hurst functions.')
            return(0)

    def CheckValidity(self):
        """Check the validity of field.

        :returns: True if attributes are properly defined, and false otherwise.
        :rtype: boolean
        """
        valid = isinstance(self.topo, perfunction)\
            and isinstance(self.hurst, perfunction)

        if valid is False:
            print("The field is not properly defined.")

        return(valid)

    def DisplayParameters(self, nfig=1):
        """Plot the graph of the topothesy and Hurst functions of the field,
        and the :term:`semi-variogram` (if available).

        :param nfig: The index of the figure. Default to 1.
        :type nfig: int, optional
        """
        if self.CheckValidity():
            self.topo.Display(nfig)
            self.hurst.Display(nfig+1)
        else:
            return(0)

        if isinstance(self.vario, sdata):
            self.vario.Display(nfig+2)

    def ComputeSemiVariogram(self, lags):
        """Compute values of the :term:`semi-variogram` of the field
        at points given by lags.

        :param lags: Lags at which to compute the semi-variogram.
        :type lags: :ref:`coordinates<coordinates>`

        :returns: Attribute vario.

        .. warning:

            This function might be slow and inaccurate when the Hurst function
            or topothesy function is not a step function.
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
            t = linspace(- pi / 2, pi / 2, 1000)
        else:
            # Determine the common step intervals of the topothesy and Hurst
            # functions.
            t = concatenate((array([-pi / 2]), c.finter[0],
                             h.finter[0], array([pi / 2])), axis=0)
            t = t[nonzero(t <= pi / 2)]
            t = t[nonzero(t >= -pi / 2)]
            t = unique(t)

        # Evaluate the topotothesy and Hurst functions at positions t.
        c.Evaluate(t)
        h.Evaluate(t)

        # Create a sdata object to store the semi-variogram.
        self.vario = sdata(lags)
        self.vario.name = 'Field semi-variogram.'

        coord2 = power(lags.xy[:, 0], 2) + power(lags.xy[:, 1], 2)
        coord2 = reshape(coord2, (coord2.size, 1))
        N = lags.N

        for k in range(c.t.size - 1):
            H = h.values[0, k]
            C = c.values[0, k]
            if C != 0 and H != 0:
                self.vario.values = self.vario.values +\
                    C * pow(2, (2 * H - 1)) / pow(N, 2 * H) *\
                    BETA_H(lags, c.t[0, k], c.t[0, k+1], H) *\
                    power(coord2, H)

    def FindOrder(self):
        """Find the order of the :term:`intrinsic` field.

        :returns: Attribute order.
        :rtype: int.

        .. warning::

            This function is only available when the Hurst function
            is a step function.
        """
        if self.CheckValidity() is False:
            return(0)

        if 'step' in self.hurst.ftype:
            inter = linspace(-pi/2, pi/2, 10000)
            self.hurst.Evaluate(inter)
            self.topo.Evaluate(inter)
            ind = nonzero(self.topo.values[:] != 0)
            self.order = ceil(amax(self.hurst.values[ind], axis=None)) - 1
        else:
            print('FindOrder(): only available for Hurst step function')
            return(0)

    def ChangeOrder(self, neworder):
        """Change the order of the :term:`intrinsic` field.

        :param int neworder: The new order of the field.

        :returns: Attributes order, hurst.

        .. warning::

            This function is only available when the Hurst function is a step
            function.
        """
        if self.CheckValidity() is False:
            return(0)

        if ('step' in self.hurst.ftype):
            order0 = self.order
            self.hurst.fparam = (neworder + 1) / (order0 + 1) *\
                self.hurst.fparam
            self.FindOrder()
            return(1)
        else:
            print('ChangeOrder(): only available for Hurst step function.')
            return(0)

    def ComputeFeatures(self):
        """Compute several features of the field.

        :returns: Attributes H, hurst_argmin_length, hurst_index_aniso,
            aniso_indices_topo, aniso_indices_hurst, aniso_sharpness_topo,
            aniso_sharpness_hurst, aniso_indices_mixed1, aniso_indices_mixed2,
            aniso_sharpness_mixed1, aniso_sharpness_mixed2.
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


def BETA_H(coord, alp1, alp2, H):
    r"""Approximation of an integral useful for the computation of
    semi-variogram.

    The approximated integral is defined as:

    .. math::
        I(x, \alpha_1, \alpha_2, H) = 2^{-2H}
        \int_{\alpha_1}^{\alpha_2} \vert \cos(\arg(x) - \theta) \vert^{2H}.

    :param coord: Coordinates :math:`x`.
    :type coord: coordinates_
    :param float alp1, alp2: Angles in :math:`[-\pi/2, \pi/2]`.
    :param float H : Hurst index in :math:`(0, 1)`.

    :returns: Values of the integral for each coordinate :math:`x`.
    :rtype: :ref:`ndarray`
    """
    from afbf.utilities import sin, atan2, sign, beta, betainc

    pi2 = pi / 2
    H = H + 0.5

    theta = zeros((coord.xy.shape[0], 1))
    for i in range(theta.size):
        if coord.xy[i, 0] != 0:
            theta[i, 0] = atan2(coord.xy[i, 1], coord.xy[i, 0])
        else:
            theta[i, 0] = sign(coord.xy[i, 1]) * pi2
    s1 = sin(alp1 - theta) / 2
    s2 = sin(alp2 - theta) / 2

    G = zeros((coord.xy.shape[0], 1))
    for i in range(coord.xy.shape[0]):
        if ((alp1 - pi2 <= theta[i, 0]) & (theta[i, 0] <= alp2 - pi2)):
            G[i, 0] = betainc(H, H, 0.5 - s2[i, 0]) +\
                betainc(H, H, 0.5 - s1[i, 0])
        elif ((alp1 + pi2 <= theta[i, 0]) & (theta[i, 0] <= alp2 + pi2)):
            G[i, 0] = betainc(H, H, 0.5 + s2[i, 0]) +\
                betainc(H, H, 0.5 + s1[i, 0])
        else:
            G[i, 0] = abs(betainc(H, H, 0.5 - s2[i, 0]) -
                          betainc(H, H, 0.5 - s1[i, 0]))

    return G * beta(H, H)
