#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for the simulation of anisotropic fractional Brownian fields.
"""

from afbf.utilities import pi, linspace, zeros, tan, atan2, cos, sin, unique
from afbf.utilities import log, nonzero, floor, absolute, arange, amax, amin
from afbf.utilities import sqrt, diff, plt, sum
from afbf import coordinates, sdata, process, perfunction, field
from afbf.utilities import pickle


class tbparameters:
    """
    Handle parameters of the turning band simulation method.

    Reference:

    [1] H. Biermé, L. Moisan and F. Richard. A turning-band method for the
        simulation of anisotropic fractional Brownian fields, Journal of
        Computational and Graphical Statistics, 24(3):885-904, 2015.
    """

    def __init__(self, K=500):
        """
        Set parameters that define the turning bands.

        Parameters
        ----------
        K : int, optional
            The approximate number of turning bands.
            The default is 500.
            This parameter determines the simulation accuracy which is about
            pi/K (in radians). By default, it is set to 500.
        """
        # Compute the angles defining the turning bands.
        self.K0 = K
        self.OptimalAngles(K)
        self.Precision()
        self.SimulationCost()

    def QuasiUniformAngles(self, K=100000):
        """
        Build a set S of K angles which are approximately uniform
        on the interval [-pi/2,pi/2] and with tangents satisfying
        tan(s)=p/q, where p is in Z and q in N*.
        This function is used to define the angles of the turning bands.

        Parameters
        ----------
        K : int, optional
            A number of angles. The default is 100000.

        Returns
        -------
        self.Kangle: list of angles in S.
        self.Pangle, self.Qangle: integer pairs
        defining angle tangents tg(s) = q / p.
        """

        from math import floor as mathfloor

        # An auxiliary function.
        def rat(x, prec):
            """
            Computes the approximation of a real number by a rational one.

            Parameters
            ----------
                x: the real number to be approximated.
                prec: the precision at which it is approximated.

            Returns
            -------
                a1, b1: numerator and denominator of the rational number
                that approximates x.
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
            Kangle[m] = atan2(p, q)
            Pangle[m] = p
            Qangle[m] = q

        Pangle[0] = 0
        Qangle[0] = 1
        Pangle[-1] = 0
        Qangle[-1] = 1

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
        """
        Apply a dynamic programming algorithm to find an optimal subset S'
        of angles among a set S of possible angles whose tangents are
        rational. S is defined as a finite subset of
        {s in [-pi/2,pi/2], tg(s)=q/p, q in Z, p in N with p^q=1}.
        To each s in S, we associate a cost C(s)=abs(p)+q, where tg(s)=p/q.
        The optimal subset S' minimizes the sum over s in S' of C(s) under
        the constraint that the difference between two successive angles
        of S' are below the precision prec.
        This function is used to define the angles of the turning bands.

        Parameters
        ----------
            N: determines the accuracy of the highest difference
            between angles of two successive bands (expressed in radians):
            if N < 1, accuracy = N else accuracy (in radians) = pi / N.

        Returns
        -------
            Kangle: list of angles in the optimal set.
            Pangle, Qangle: integer pairs defining angle tangents tg(s)=q/p.
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

    def PlotBands(self):
        """
        Plot the angles used to define the turning bands.
        """
        plt.figure(1)
        plt.subplot(211)
        for i in range(0, self.Kangle.size):
            plt.plot([0, cos(self.Kangle[i])], [0, sin(self.Kangle[i])], 'r-')
            plt.plot(cos(self.Kangle[i]), sin(self.Kangle[i]), 'kx')
        plt.axis([-1, 1, -1, 1])
        plt.xlabel(r"$\cos(\phi)$")
        plt.ylabel(r"$\sin(\phi)$")
        plt.show()
        plt.subplot(212)
        plt.plot(arange(1, self.Kangle.size+1, 1).reshape(self.Kangle.shape),
                 self.Kangle, 'rx-')
        plt.axis([1, self.Kangle.size, -pi/2, pi/2])
        plt.xlabel(r"$i$")
        plt.ylabel(r"$\phi_i$")
        plt.show()

    def SimulationCost(self):
        """
        Compute the computational cost associated to the turning bands.
        """
        v = absolute(self.Pangle) + self.Qangle
        v = v * log(v)
        self.cost = sum(v)

    def Precision(self):
        """
        Compute the precision of the turning bands.
        """
        self.acc = amax(diff(self.Kangle))

    def DisplayInformation(self):
        """
        Display information about simulation.
        """
        print('Turning band information.')
        print('Number of bands :' + str(self.Kangle.size))
        print('Precision       :' + str(self.acc))
        print('Simulation cost :' + str(self.cost))

    def Save(self, filename):
        """
        Save the turning band parameters.

        Parameters
        ----------
        filename : str
            File name (without extension).
        """
        with open(filename + '.pickle', 'wb') as f:
            pickle.dump(self.K0, f)


class tbfield(field):
    r"""
    Handle turning band fields that can be simulated. This class herits
    from the class field.

    ..math::Given a set $\{ \phi_k, k =1,\cdots, K\}$ composed of angles in
            $[-\pi/2,\pi/2)$, two $\pi$-periodic functions $c$ and $h$,
            a discrete AFBF $X$ is defined as
            $$X(y) = \sum_{k=1}^K c(\phi_k) Z_k(\langle \phi_k, y \rangle),$$
            where $Z_k$ are independents fractional Brownian motions of
            Hurst index $h(\phi_k)$.
    """

    def __init__(self, fname='fbf', topo=None, hurst=None, K=500):
        """
        Set the model, including the topothesy and hurst functions and the
        turning bands.
        See SetModel and InitTurningBands.

        Parameters
        ----------
        fname : str, optional
            Name of the field. The default is 'fbf'.
        topo, hurst : perfunction, optional
            Topothesy and Hurst functions. The default is None.
        K : int or tbparameters, optional
            Number of bands. The default is 500.
            if K is of the class tbparameters, the bands are given directly
            in K.
        """
        self.SetModel(fname, topo, hurst)
        if isinstance(K, tbparameters):
            self.tb = K
        else:
            self.InitTurningBands(K)

    def InitTurningBands(self, K):
        """
        Initialize the parameters of turning bands that define the field,
        and are used to simulate the field.

        Parameters
        ----------
            K: int
                Number of bands.
        """
        # Compute the optimal angles of turning bands.
        self.tb = tbparameters(K)
        # Evaluate the topothesy and Hurst functions of the fields
        # at angles of the turning bands.
        if self.CheckValidity():
            self.EvaluateTurningBandParameters()

    def CheckBands(self):
        """
        Check the validity of bands (coarse test).
        """
        valid = isinstance(self.tb, tbparameters)
        if valid is False:
            print("TurningBands: the bands are not defined properly.")

        return(valid)

    def EvaluateTurningBandParameters(self):
        """
        Evaluate the field parameters at angles of the turning bands.
        """
        if self.CheckValidity() and self.CheckBands():
            self.topo.Evaluate(self.tb.Kangle)
            self.hurst.Evaluate(self.tb.Kangle)

    def ComputeApproximateSemiVariogram(self, lags, logvario=0, evaluate=True):
        """
        Compute the semi-variogram of the turning-band field.

        Parameters
        ----------
            lags: coordinates
                lags at which to compute the semi-variogram.
                logvario: int
            logvario: int, optional
                If log>0, a log semi-variogram is computed instead of the
                semi-variogram. The default is 0.
            evaluate: boolean, optional
                if evaluate is True the topothesy and hurst functions
                are evaluated at bands. If not, the current values are used.
                The default is True.
        Returns
        -------
            self.svario: sdata
                The semi variogram of the turning-band field.
        """
        self.logvario = logvario
        # Evaluate the semi-variogram.
        self.Simulate(lags, True, evaluate)

    def ExtendTopothesy(self, M1, M2):
        """
        Extend the definition of the topothesy of the field.
        """
        self.extended = True
        self.gtopo = []
        for m in range(M1 * 2):
            self.gtopo.append(perfunction('step-smooth', M2))

    def SaveAll(self, filename):
        """
        Save the model in file.

        Parameters
        ----------
        filename : str
            File name without extension.
        """
        self.Save(filename)
        self.tb.Save(filename + '-tb')

    def Simulate(self, coord=None, vario=False, evaluate=True):
        """
        Simulation of anisotropic fractional Brownian fields by a
        turning-band method.

        Reference:

        [1] H. Biermé, L. Moisan and F. Richard. A turning-band method for the
            simulation of anisotropic fractional Brownian fields, Journal of
            Computational and Graphical Statistics, 24(3):885-904, 2015.

        Parameters
        ----------
            coord: coordinates
                coordinates where to simulate the field or evaluate its
                semi-variogram. The default is None, indicating that
                coordinates will be set automatically.
            vario: boolean
                if vario = True, the function computes the semi-variogram of
                the simulation field. The default is False.
            evaluate: boolean
                if evaluate is True the topothesy and hurst functions
                are evaluated at bands. If not, the current values are used.
                The default is True.

        Returns
        -------
            X: sdata
            a simulation of the field or the semi-variogram
            of the simulation field (if vario=True).

        Example:

        >>> print('Simulation of a fractional Brownian field.')
        >>> from afbf import tbfield
        >>> # Definition of a fractional Brownian field.
        >>> field = tbfield()
        >>> field.DisplayParameters(1)
        >>> field.tb.DisplayInformation()
        >>> # Simulation of the field.
        >>> X = field.Simulate()
        >>> X.Display(3)
        """
        def Weight_TB(q, h, c, ang0, ang1):
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
        if evaluate:
            self.EvaluateTurningBandParameters()
        if coord is None:
            coord = coordinates(512)

        X = zeros((coord.ncoord, 1))

        C = self.topo.values
        H = self.hurst.values
        H0 = floor(H)
        H = H - H0

        M = amax(coord.xy, axis=None)
        N = coord.N

        fbm = process('fbm')  # Fractional Brownian motion on bands.
        if self.extended:
            fbm.ExtendFBM(len(self.gtopo) / 2)
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
                                        self.tb.Kangle[k-1], kang)

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
                        for m in range(1, len(self.gtopo)):
                            self.gtopo[m].Evaluate(kang)
                            fparam[0, m + 1] = self.gtopo[m].values[0, 0]
                            fparam[0, 0] = sum(absolute(fparam), axis=None)

                    # Simulate a realization of increments of the FBM.
                    fbm.Simulate_CirculantCovarianceMethod(nbsamp)
                    # Integrate the increments to obtain a FBM realization.
                    fbm.IntegrateProcess(H0[0, k] + 1)
                    # Update of the simulation field.
                    fbm.y = fbm.y[indr, 0] - fbm.y[ind0, 0]
                    X = X + sqrt(weig0) * pow(weig, H[0, k]) *\
                        fbm.y.reshape(fbm.y.size, 1)

        if vario:
            self.svario = sdata()
            self.svario.coord = coord
            self.svario.name = 'Semi-variogram of the simulation field.'
            self.svario.values = X
            return 1
        else:
            sfield = sdata()
            sfield.coord = coord
            sfield.name = 'Field simulation.'
            sfield.values = X
            return sfield


if __name__ == '__main__':
    import doctest
    doctest.testmod()
