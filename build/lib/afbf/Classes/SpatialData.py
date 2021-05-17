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
"""Module for the management of spatial data.

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>
"""

from afbf.utilities import reshape, arange, repmat, concatenate, array, zeros
from afbf.utilities import amin, amax, mean, matmul, floor, linspace, sign
from afbf.utilities import power, plt
from afbf.utilities import make_axes_locatable, ndarray


class coordinates:
    r"""This class handles a set of coordinates in the plane.

    .. _coordinates:

    Coordinates are pairs (x, y) of integers referring to a plane position
    (x / N, y / N). Set of coordinates can either be on a uniform grid
    or at arbitrary positions.

    A uniform grid is defined as :math:`[\![1, N]\!] \times  [\![1, N]\!]`.
    It can also be signed, in which case it is of the form
    :math:`[\![1, N]\!] \times  [\![-N, N]\!]`.

    :example:

        Define and display a grid of size 10 x 10.

    .. code-block:: python

        from afbf import coordinates
        coord = coordinates(10)
        coord.Display(1)

    .. image:: Figures/grid.png

    :param xy: set of Cartesian coordinates; xy[m, :] are the mth coordinates.
    :type xy: :ref:`ndarray` of shape (ncoord, 2)

    :param int N: Factor to be applied to coordinates.

    :param int Nx: Grid dimension: number of x coordinates.

    :param int Ny: Grid dimension: number of y coordinates.

    :param bool grid: True if grid coordinates.

    """

    def __init__(self, N=-1):
        """Constructor method.

        :param N: The size of the grid. Default to -1.
        :type N: int, optional
            If a grid is to be created, set N to a positive integer.
        """
        if isinstance(N, int) and N > 0:
            self.DefineUniformGrid(N)
        else:
            self.xy = None  # cartesian coordinates.
            self.Nx = None  # grid dimension: number of x coordinates.
            self.Ny = None  # grid dimension: number of y coordinates.
            self.N = None  # coordinate value corresponding to position 1.
            self.grid = False  # type of coordinates.

    def CheckValidity(self):
        """Check the validity of coordinates.

        :returns: True if attributes are properly defined, and false otherwise.
        :rtype: boolean
        """
        valid = isinstance(self.xy, ndarray) and self.xy.shape[1] == 2
        if valid is not True:
            print("Coordinates are not properly defined.")

        return(valid)

    def DefineUniformGrid(self, N, step=1, signed=False):
        r"""Define a uniform grid.

        :param int N: number of x- and y- coordinates.

        :param step: step between grid points. Default to 1.
        :type step: int, optional

        :param signed: True if the grid is to be signed.
        :type signed: boolean, optional

        :returns: Attributes xy, N, Nx, Ny, grid.
        """
        self.N = N

        vx = arange(1, N+1, step, dtype=int)
        if signed:
            vy = arange(-N, N+1, step, dtype=int)
        else:
            vy = vx

        Nx = vx.size
        Ny = vy.size
        x = repmat(vx.reshape(1, Nx), Ny, 1)
        vy = vy[::-1]
        y = repmat(vy.reshape(Ny, 1), 1, Nx)

        # Coordinates.
        N2 = Nx * Ny
        self.xy = concatenate((x.reshape(N2, 1), y.reshape(N2, 1)),
                              axis=1)
        self.grid = True
        self.Nx = Nx  # grid dimension.
        self.Ny = Ny

    def DefineNonUniformLocations(self, xy):
        """Import a set of positions provided in an array.

        :param xy:
            An array of size (ncoord, 2) containing
            coordinates: xy[m, :] are the mth coordinates.
        :type xy: :ref:`ndarray`

        :returns: Attributes xy, M, N, grid.

        :example: Define coordinates at some given positions.

        .. code-block:: python

            from afbf import coordinates
            from numpy import array

            xy = array([[1, 2], [3, 4], [-2, 2], [5, 6]], dtype=int)
            coord = coordinates()
            coord.DefineNoneUniformLocations(xy)

        """
        if isinstance(xy, ndarray) and xy.shape[1] == 2:
            self.xy = zeros(xy.shape, dtype=int)
            self.xy[:] = xy[:]
            self.N = amax(self.xy, axis=None)
            self.grid = False
        else:
            print("DefineNonUniform...: provide xy as an ndarray (M, 2).")

    def ApplyAffineTransform(self, A):
        r"""Apply an affine transform A to coordinates.

        Given a matrix :math:`A` of shape (2, 2) and coordinates
        :math:`(x, y)`,
        the transform is defined as

        .. math::
            (\tilde x, \tilde y) = (x, y) A.

        :param A:
            An array of shape (2, 2) of type int defining the
            affine transform.
        :type A: :ref:`ndarray`

        :returns: Attributes xy.

        :example: Apply an affine transform to a uniform grid.

        .. code-block:: python

            from afbf import coordinates
            from numpy import array

            coord = coordinates(10)
            coord.Display(1)
            A = array([[1, 3], [2, 1]], dtype=int)
            coord.ApplyAffineTransform(A)
            coord.Display(2)

        .. image:: ./Figures/coordaff.png

        """
        if self.CheckValidity() is False:
            return(0)

        if isinstance(A, ndarray) and A.shape[0] == 2 and A.shape[1] == 2:
            self.xy = matmul(self.xy, A)
            self.N = amax(self.xy, axis=None)
        else:
            print("ApplyAffine...: provide A as ndarray of size (2, 2).")

    def ProjectOnAxis(self, u, v):
        r"""Project coordinates on an axis oriented in (u, v).

        Given the axis :math:`(u, v)` and coordinates :math:`(x, y)`,
        the projection is given by

        .. math::
            z = u\: x + v \: y

        :param int u, v: The projection axis.

        :returns: Projection coordinates.
        :rtype: :ref:`ndarray` of shape (ncoord, 2)
        """
        if self.CheckValidity() is False:
            return(0)

        ind = u * self.xy[:, 0] + v * self.xy[:, 1]

        return ind.reshape((ind.size, 1))

    def Display(self, nfig=1):
        """
        Display the positions given by coordinates.

        :param int nfig: The index of the figure. Default to 1.
        :type nfig: int, optional
        """
        if self.CheckValidity() is False:
            return(0)

        plt.figure(nfig)
        plt.plot(self.xy[:, 0] / self.N, self.xy[:, 1] / self.N, 'rx')
        plt.axis('equal')


class sdata:
    """This class handles spatial data.

    .. _sdata:

    Spatial data includes but are not restricted to images.
    Images are particular spatial data defined on a uniform grid.

    :param coordinates coord: Coordinates where data are defined.

    :param values: Spatial values observed at each position of coord;
        values[m] is the value observed at position coord[m, :].
    :type values: :ref:`ndarray`

    :param :ref:`ndarray` M: size of the image (number of rows, columns).

    :param name: Name of data. Default to 'undefined'.
    :type name: str, optional
    """

    def __init__(self, coord=None, name='undefined'):
        """Contructor method.

        :param coord:
            A set of coordinates where the data is defined.
            The default is None.
        :type coord: :ref:`coordinates<coordinates>`

        :param name: Name.
        :type name: str, optional

        :returns: Attributes values, coord, name.
        """
        self.name = name
        self.M = None
        if coord is not None:
            if isinstance(coord, coordinates):
                self.coord = coord
                if coord.grid:
                    self.M = array([coord.Ny, coord.Nx])
                    self.name = 'Image'
                self.values = zeros((coord.xy.shape[0], 1))
            else:
                print("sdata.__init__: provide coord as coordinates.")
                return(None)
        else:
            self.coord = None
            self.values = None

    def Display(self, nfig=1):
        """Display an image.

        :param nfig: Figure index. Default to 1.
        :type nfig: int, optional
        """
        if self.coord.grid:
            M = self.M
            N = self.coord.N
            plt.figure(nfig)
            fig = plt.subplot()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(self.name)
            v = plt.imshow(self.values.reshape(M),
                           origin='upper', cmap='gray')
            r = M[1] / M[0]
            crx = int(max(1, floor(5 * min(1, r))))
            cry = int(max(1, floor(5 * min(1, 1 / r))))
            if self.coord.xy is None:
                cx = linspace(0, M[1] / N, crx)
                cy = linspace(0, M[0] / N, cry)
            else:
                cx = linspace(amin(self.coord.xy[:, 0]) / N,
                              amax(self.coord.xy[:, 0]) / N, crx)
                cy = linspace(amin(self.coord.xy[:, 1]) / N,
                              amax(self.coord.xy[:, 1]) / N, cry)
            cy = cy[::-1]

            plt.xticks(linspace(0, M[1], crx), floor(cx * 1000) / 1000)
            plt.yticks(linspace(0, M[0], cry), floor(cy * 1000) / 1000)
            plt.ticklabel_format
            divider = make_axes_locatable(fig)
            cax = divider.append_axes("right", size="5%", pad=0.6)
            plt.colorbar(v, cax=cax)
            plt.show()
        else:
            print('SpatialData.Display: not available on non-uniform sites.')

    def CreateImage(self, M):
        """Create an image.

        :param M:
            An array of size 2 giving the number of rows and columns of
            the matrix.

        :type M: :ref:`ndarray`

        :returns: Attribute coord.
        """
        if isinstance(M, ndarray) and M.size == 2:
            self.coord = coordinates()
            self.M = reshape(M, 2)
            self.coord.N = amax(M)
            self.coord.Nx = M[1]
            self.coord.Ny = M[0]
            self.coord.grid = True
        else:
            print('CreateImage: size of image should be an array of size 2.')
            return(0)

    def ComputeIncrements(self, hx, hy, order=0):
        r"""Compute increments of an image.

        Given some lags :math:`(h_x, h_y)` and
        an order :math:`J`, increments :math:`Z` of order :math:`J`
        of the image form an image defined in a recursive way by

        .. math::
            \left\{ \begin{array}{l}
            X^{(0)} = X, \\
            X^{(j+1)} [x, y] = X^{(j)}[x, y] - X^{(j)}[x - h_x, y - h_y],
            \:\: \mathrm{for} \:\: j = 0, \cdots, J, \\
            Z = X^{(J+1)}.
            \end{array}\right.

        :param int hx, hy:
            Horizontal and vertical lags.

        :param int order:
            Order of the increment. The default value is 0.

        :returns:
            The increment image.
        :rtype: sdata
        """
        hx = int(hx)
        hy = int(hy)

        if not self.coord.grid:
            print('sdata.ComputeIncrements:  only applies to an image.')
            return(0)

        N = self.coord.N

        if (abs(hx) >= self.M[1]) or (abs(hy) >= self.M[0]):
            print('ComputeIncrements: lags are too large for image size.')
            return(0)

        valincre = reshape(self.values, self.M)

        order = order + 1
        for j in range(order):
            M = valincre.shape
            hxa = int(abs(hx))
            hya = int(abs(hy))
            # Compute Z(x, y) - Z(x - hx, y - hy).
            # The computation is expected in Euclidien coordinates, but the
            # values of Z are stored in a matrix. The term Z[i, j] of this
            # matrix corresponds to Z(x, y) with x = j and y = M[0] - i,
            # M[0] being the number of rows.

            if (hx >= 0) and (hy >= 0):
                valincre = valincre[0:(M[0] - hya), hxa:M[1]]\
                    - valincre[hya:M[0], 0:(M[1] - hxa)]
            elif hx >= 0:  # hx >=0  and hy < 0
                valincre = valincre[hya:M[0], hxa:M[1]]\
                    - valincre[0:(M[0] - hya), 0:(M[1] - hxa)]
            elif hy >= 0:  # hx < 0 and hy >= 0
                valincre = valincre[0:(M[0] - hya), 0:(M[1] - hxa)]\
                    - valincre[hya:M[0], hxa:M[1]]
            else:  # hx < 0 and hy < 0
                valincre = valincre[hya:M[0], 0:(M[1] - hxa)]\
                    - valincre[0:(M[0] - hya), hxa:M[1]]

        incre = sdata()
        M = array(valincre.shape)
        incre.CreateImage(M)
        incre.values = valincre
        incre.coord.N = N
        incre.name = 'Increments'

        return(incre)

    def ComputeLaplacian(self, scale=1):
        """Compute the discrete laplacian of an image.

        Given some scale :math:`s`, the Laplacian :math:`Z` of the image
        is an image defined as

        .. math::
            Z[x, y] = 4 X[x, y]-X[x-s, y]-X[x+s, y]- X[x, y-s] - X[x, y+s].

        :param scale:
            Scale at which the Laplacian is computed. Default to 1.
        :type scale: int , optional

        :returns:
            The image Laplacian.
        :rtype: sdata
        """
        if not self.coord.grid:
            print('data.ComputeLaplacian:  only applies to an image.')
            return(0)

        # Second-order discrete derivative with respect to x and y variables.
        dx2 = self.ComputeIncrements(scale, 0, 1)
        dy2 = self.ComputeIncrements(0, scale, 1)
        Mdx2 = dx2.M
        Mdy2 = dy2.M
        mx = min(Mdx2[1], Mdy2[1])
        my = min(Mdx2[0], Mdy2[0])

        laplacian = sdata()
        M = array([my, mx])
        laplacian.CreateImage(M)
        laplacian.coord.N = self.coord.N
        laplacian.name = 'Image Laplacian.'
        laplacian.values =\
            dx2.values.reshape(Mdx2)[0:my, 0:mx] +\
            dy2.values.reshape(Mdy2)[0:my, 0:mx]

        return(laplacian)

    def ComputeImageSign(self):
        """Compute the sign of an image.

        :returns:
            The sign image.
        :rtype: sdata
        """
        if not self.coord.grid:
            print('data.ComputeImageSign:  only applies to an image.')
            return(0)

        simage = sdata()
        simage.CreateImage(self.M)
        simage.values = sign(self.values)
        simage.coord.N = self.coord.N
        simage.name = 'Image sign.'

        return simage

    def ComputeQuadraticVariations(self, lags, order=0):
        """Compute the quadratic variations of an image.

        :param lags: Lags at which to compute quadratic variations.
        :type lags:  coordinates_

        :aram order: The order of the quadratic variations. The default is 0.
        :type order: int, optional

        :returns: Quadratic variations.
        :rtype: sdata

        .. note::

            This method only applies to an image.
        """
        if not self.coord.grid:
            print('ComputeQuadraticVariations:  only applies to an image.')
            return(0)

        if not isinstance(lags, coordinates):
            print('ComputeQuadraticVariations: provide lags as coordinates.')
            return(None)

        qvar = sdata(lags)
        qvar.name = 'Quadratic variations.'

        for j in range(lags.xy.shape[0]):
            hx = lags.xy[j, 0]
            hy = lags.xy[j, 1]

            # Compute field increments.
            incre = self.ComputeIncrements(hx, hy, order)
            values = incre.values
            # Compute the quadratic variations.
            qvar.values[j, 0] = mean(power(values, 2), axis=None)

        return(qvar)

    def ComputeEmpiricalSemiVariogram(self, lags):
        """Compute the empirical semi-variogram of an image.

        :param lags: Lags at which to compute quadratic variations.
        :type lags: coordinates_

        :returns: The semi-variogram.
        :rtype: sdata

        .. note::

            This method only applies to an image.

        """
        if not self.coord.grid:
            print('ComputeEmpiricalSemiVariogram: only applies to an image.')
            return(0)

        # Compute quadratic variations
        evario = self.ComputeQuadraticVariations(lags)
        evario.name = 'Empirical semi-variogram.'

        # Compute the semi-variogram.
        evario.values[:, 0] = 0.5 * evario.values[:, 0]

        return(evario)
