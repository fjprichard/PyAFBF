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

r"""Module for the management of turning band fields.

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>
"""

from afbf.utilities import pi, linspace, zeros, tan, arctan2, cos, unique
from afbf.utilities import log, nonzero, floor, absolute, amax, amin
from afbf.utilities import sqrt, diff, sum, array, concatenate
from afbf import coordinates, sdata, process, perfunction, field


class tbfield(field):
    r"""This class handles the turning-band fields.

    They are defined as follows.
    Let :math:`\{ \phi_k, k =1,\cdots, K\}` be a set of ordered angles in
    :math:`[-\pi/2,\pi/2)`, called turning bands. Let :math:`\tau`
    and :math:`\beta` be the topothesy and Hurst functions
    of an AFBF.

    .. math::
        W(y) = \sum_{k=1}^K \lambda_k X_k(\langle \phi_k, y \rangle),

    where :math:`\lambda_k` are coefficients depending on :math:`\tau(\phi_k)`
    and :math:`X_k` are independent fractional Brownian motions of
    Hurst index :math:`\beta(\phi_k)`.

    Turning band fields are used for simulation.

    :example:

        Simulation of a fractional Brownian field.

    >>> from afbf import tbfield
    >>> # Definition of a fractional Brownian field.
    >>> field = tbfield()
    >>> field.DisplayParameters(1)
    >>> field.tb.DisplayInformation()
    >>> # Simulation of the field.
    >>> X = field.Simulate()
    >>> X.Display(3)

    :param tb: Turning band parameters of the fields.
    :type tb: :ref:`tbparameters<tbparameters>`

    :param svario: The semi variogram of the turning-band field.
    :type svario: :ref:`sdata<sdata>`
    """

    def __init__(self, fname='fbf', topo=None, hurst=None, K=500):
        """Constructor method.

        :param fname:  Name of the field. The default is 'fbf'.
        :type fname: str, optional

        :param topo: The topothesy function of the field.
        :type topo: :ref:`perfunction<perfunction>`, optional

        :param hurst: The Hurst function of the field.
        :type hurst: :ref:`perfunction<perfunction>`, optional

        :param K: Number of bands. The default is 500.
            if K is of the class :ref:`tbparameters<tbparameters>`,
            the bands are given directly in K.
        :type K: int or :ref:`tbparameters<tbparameters>`

        :returns: Attributes fname, topo, hurst and tb.

        .. seealso::
            Constructor method of the class :ref:`field`.
        """
        self.SetModel(fname, topo, hurst)
        if isinstance(K, tbparameters):
            self.tb = K
        else:
            self.InitTurningBands(K)

    def InitTurningBands(self, K):
        """Set parameters of turning bands.

        :param int K: The number of bands.

        :returns: Attribute tb.
        """
        # Compute the optimal angles of turning bands.
        self.tb = tbparameters(K)
        # Evaluate the topothesy and Hurst functions of the fields
        # at angles of the turning bands.
        if self.CheckValidity():
            self.EvaluateTurningBandParameters()

    def EvaluateTurningBandParameters(self):
        """Evaluate the topothesy and Hurst function at turning band angles.

        :returns: Attributes topo and hurst.
        """
        if self.CheckValidity():
            self.topo.Evaluate(self.tb.Kangle)
            self.hurst.Evaluate(self.tb.Kangle)

    def ComputeApproximateSemiVariogram(self, lags, logvario=0, evaluate=True):
        """Compute the semi-variogram of the turning-band field.

        :param lags: Lags at which to compute the semi-variogram.
        :type lags: :ref:`coordinates<coordinates>`

        :param logvario:
            If log>0, a log semi-variogram is computed instead
            of the semi-variogram. The default is 0.
        :type logvario: int, optional

        :param evaluate:
            if True the topothesy and hurst functions
            are evaluated at bands. If not, the current values are used.
            The default is True.
        :type evaluate: boolean, optional

        :return: Attribute svario.
        """
        self.logvario = logvario
        # Evaluate the semi-variogram.
        self.Simulate(lags, True, evaluate)

    def ExtendTopothesy(self, M1=2, M2=4):
        """Extend the definition of the topothesy of the field.

        :param int M1: Number of spectrum subdivisions.
        :param int M2: Number of angle subdivisions.
        """
        self.extended = True
        self.gtopo = []
        for m in range(M1):
            self.gtopo.append(perfunction('step-smooth', M2))

    def Simulate(self, coord=None, vario=False, evaluate=True):
        """Simulate the anisotropic fractional Brownian fields by a
        turning-band method.

        .. note::
            The method was designed by H. Biermé, L. Moisan
            and F. Richard. It is described in :cite:p:`Bierme-2015-TBM`.

        :param coord:
                Coordinates where to simulate the field or evaluate its
                semi-variogram. The default is None, indicating that
                coordinates will be set automatically.
        :type coord: :ref:`coordinates<coordinates>`

        :param vario:
                if True, the function computes the semi-variogram of
                the simulation field. The default is False.
        :type vario: boolean, optional

        :param evaluate:
                if True, the topothesy and hurst functions
                are evaluated at bands. If not, the current values are used.
                The default is True.
        :type evaluate: boolean, optional

        :returns:
            A simulation of the field or the semi-variogram
            of the simulation field (if vario=True).
        :rtype: :ref:`sdata<sdata>`
        """
        def Weight_TB(q, h, c, ang0, ang1, N):
            """
            Compute the weight of a turning band using a rectangular integral
            approximation.
            """
            weig0 = (ang1 - ang0) * c
            if q != 0:
                weig = cos(ang1) / (q * N)
            else:
                weig = 1. / N

            return weig0, weig

        if self.CheckValidity() is False:
            return(0)

        if coord is None:
            coord = coordinates(512)

        if isinstance(coord, coordinates):
            N = coord.N
            if not coord.grid:
                # Add the frame corner locations.
                coord.xy = concatenate((coord.xy,
                                        array([[1, N], [N, 1],
                                               [1, 1], [N, N]])),
                                       axis=0)
            M = amax(coord.xy, axis=None)
        else:
            print('Simulate: provide coord as coordinates.')

        if evaluate:
            self.EvaluateTurningBandParameters()

        X = zeros((coord.xy.shape[0], 1))

        C = self.topo.values
        H = self.hurst.values
        H0 = floor(H)
        H = H - H0

        fbm = process('fbm')  # Fractional Brownian motion on bands.
        if self.extended:
            fbm.ExtendFBM(len(self.gtopo))
            fparam = fbm.gf.fparam
            fbm.gn = 2 * amax(abs(self.tb.Pangle) + self.tb.Qangle) * M

        for k in range(1, self.tb.Kangle.size):
            kang = self.tb.Kangle[k]
            p = self.tb.Pangle[k]
            q = self.tb.Qangle[k]

            # Number of samples to be simulated on the turning band.
            nbsamp = int(M * (abs(p) + q) + 1)

            if C[0, k] != 0 and H[0, k] != 0:
                # Weight the process on the turning band (based on rectangular
                # integral approximation of the variogram).
                weig0, weig = Weight_TB(q, H[0, k], C[0, k],
                                        self.tb.Kangle[k-1], kang, N)

                # Projection of grid coordinates on the band.
                indr = coord.ProjectOnAxis(q, p)

                # Definition of a fBm of Hurst index H[k].
                fbm.param = H[0, k]

                if vario:
                    # Computation of the semi-variogram of the band process.
                    fbm.ComputeSemiVariogram(indr * weig, self.logvario)

                    # Computation of the simulation field variogram.
                    X = X + weig0 * fbm.vario
                else:
                    ind0 = min(0, amin(indr))
                    indr = indr - ind0

                    if fbm.extended:
                        # Definition of generalized coefficients.
                        for m in range(len(self.gtopo)):
                            self.gtopo[m].Evaluate(array([kang]))
                            fparam[0, m] = self.gtopo[m].values[0, 0]

                    # Simulate a realization of increments of the FBM.
                    fbm.Simulate_CirculantCovarianceMethod(nbsamp)
                    # Integrate the increments to obtain a FBM realization.
                    fbm.IntegrateProcess(H0[0, k] + 1)
                    # Update of the simulation field.
                    fbm.y = fbm.y[indr, 0] - fbm.y[ind0, 0]
                    X = X + sqrt(weig0) * pow(weig, H[0, k]) *\
                        fbm.y.reshape(fbm.y.size, 1)

        if vario:
            self.svario = sdata(coord)
            self.svario.name = 'Semi-variogram of the simulation field.'
            self.svario.values = X
            return(1)
        else:
            if not coord.grid:
                coord.xy = coord.xy[0:-4, :]
                X = X[0:-4, :]
            sfield = sdata(coord)
            sfield.name = 'Field simulation.'
            sfield.values = X
            return(sfield)


class tbparameters:
    r"""This class handles parameters of the turning band field.

    .. _tbparameters:

    :param int K: the number of bands.

    :param Kangle:
        Angles of the turning bands.
        The tangent of each angle :math:`\varphi` have a tangent which
        satisfies

        .. math:: \tan(\varphi) = \frac{p}{q},

        for some :math:`p \in \mathbb{Z}` and :math:`q \in \mathbb{N}^*`.
    :type Kangle: :ref:`ndarray`

    :param Pangle: Denominators :math:`q` of angle tangents.
    :type Pangle: :ref:`ndarray`

    :param Qangle: Numerators :math:`p` of angle tangents.
    :type Qangle: :ref:`ndarray`

    :param scalar cost: Angle cost (dynamic programming).

    :param scalar acc: Precision (dynamic programming).
    """

    def __init__(self, K=500):
        r"""Constructor method.

        :param K: An approximate number of turning bands. The default is 500.
        :type K: int, optional

        .. note::
            This parameter determines the simulation accuracy which is about
            :math:`\frac{\pi}{K}` (in radians).
        """
        # Compute the angles defining the turning bands.
        self.K0 = K
        self.OptimalAngles(K)
        self.Precision()
        self.SimulationCost()

    def QuasiUniformAngles(self, K=100000):
        r"""Build a set of K angles which are approximately uniform
        on the interval :math:`[-\frac{\pi}{2},\frac{\pi}{2}]`.

        :param K:  A number of angles. The default is 100000.
        :type K: int, optional

        :returns: Attributes Kangle, Pangle, Qangle.
        """

        from math import floor as mathfloor

        # An auxiliary function.
        def rat(x, prec):
            """Computes the approximation of a real number by a rational one.

            :param scalar x: the real number to be approximated.
            :param scalar prec: the precision at which it is approximated.

            :returns: the numerator and denominator of the approximation.
            """

            a0, b0, b1 = 1, 0, 1
            e = x
            q = mathfloor(x)
            a1 = q
            while abs(x - a1 / b1) > prec:
                e = 1.0 / (e - q)
                q = mathfloor(e)
                a2 = q * a1 + a0
                b2 = q * b1 + b0
                a0, b0 = a1, b1
                a1, b1 = a2, b2

            return a1, b1

        # The main part.
        if K == 1:
            K = 2

        Kangle = linspace(-pi/2, pi/2, K)
        K = len(Kangle)
        Pangle = zeros(K, dtype=int)
        Qangle = zeros(K, dtype=int)
        delta = Kangle[1] + pi/2

        for m in range(1, K-1):
            # Precision.
            prec = delta * (1 + pow(tan(Kangle[m]), 2)) * 0.5
            # Rational approximation of the slope tan(Kangle(l))
            # of orientation Kangle(l) at precision prec.
            p, q = rat(tan(Kangle[m]), prec)
            # Update the orientation
            Kangle[m] = arctan2(p, q)
            Pangle[m] = p
            Qangle[m] = q

        Pangle[0] = -1
        Qangle[0] = 0
        Pangle[-1] = 1
        Qangle[-1] = 0

        # Removal of possible doublons.
        Kangle, ind = unique(Kangle, return_index=True)
        Pangle = Pangle[ind]
        Qangle = Qangle[ind]

        self.Kangle = Kangle
        self.Pangle = Pangle
        self.Qangle = Qangle

        self.Precision()
        self.SimulationCost()

    def OptimalAngles(self, N=500):
        r"""Compute a set of optimal angles by dynamic programming.

        :param scalar N: The expected precision.
            if :math:`N < 1`, the precision is set to N
            else the precision (in radians) is set to :math:`\frac{pi}{N}`.

        :returns: Attributes Kangle, Pangle, Qangle.

        .. note::

            The dynamic programming algorithm finds an optimal subset
            :math:`S'` of angles among a set :math:`\Phi` of possible angles
            whose tangents are rational. :math:`\Phi`is a subset of

            .. math::
                \{\varphi \in [-\pi/2,\pi/2], \tan(\varphi)=p/q, p
                \in \mathbb{Z}, q \in \mathbb{N}^\ast, p \wedge q=1\}.

            A cost :math:`C(\varphi)=\vert q \vert+p` is associated to each
            angle :math:`\varphi`.

            The optimal subset :math:`\Phi'` minimizes
            :math:`\sum_{\varphi \in \Phi'} C(\varphi)` under
            the constraint that the difference between two
            successive angles are below a given precision.

        """
        if N > 1:
            prec = pi / N
        else:
            prec = N
            N = floor(pi / N)

        Nmax = 10000
        # Definition of a set S of possible angles
        # and their associated costs.

        if N < Nmax/3:
            self.QuasiUniformAngles(Nmax)
        else:
            self.QuasiUniformAngles(10 * N)

        Cost = absolute(self.Pangle) + self.Qangle
        Cost = Cost * log(Cost)
        nvec = len(self.Kangle)

        # Dynamic programming for the selection of an optimal subset
        cost = zeros(nvec)  # Partial costs.
        pos = zeros(nvec, dtype=int)
        cost[nvec-1] = Cost[nvec-1]
        pos[nvec-1] = nvec-1
        ind = range(0, nvec-1)
        for i in ind[::-1]:
            bound = self.Kangle[i] + prec
            bestj = i + 1
            mini = cost[bestj]
            # Seek among upper angles at distance below prec
            # of the current angle for the one with a minimal partial cost.
            for j in range(i+2, nvec):
                if self.Kangle[j] > bound:
                    break
                if cost[j] < mini:
                    bestj = j
                    mini = cost[bestj]

            # Define the partial cost of the current angle
            # and the best associated upper angle.
            cost[i] = Cost[i] + mini
            pos[i] = bestj

        # Build the set S' by finding the best path.
        i = 0
        Select = zeros(nvec)
        while i < nvec-1:
            Select[i] = 1
            i = pos[i]
        Select[nvec-1] = 1

        ind = nonzero(Select == 1)
        self.Kangle = self.Kangle[ind]
        self.Pangle = self.Pangle[ind]
        self.Qangle = self.Qangle[ind]

    def SimulationCost(self):
        """Compute the computational cost associated to the turning bands.
        """
        v = absolute(self.Pangle) + self.Qangle
        v = v * log(v)
        self.cost = sum(v)

    def Precision(self):
        """Compute the precision of the turning bands.
        """
        self.acc = amax(diff(self.Kangle))

    def DisplayInformation(self):
        """Display information about simulation.
        """
        print('Turning band information.')
        print('Number of bands :' + str(self.Kangle.size))
        print('Precision       :' + str(self.acc))
        print('Simulation cost :' + str(self.cost))
