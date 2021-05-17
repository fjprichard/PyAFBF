#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for the management of spatial data.
"""

from afbf.utilities import reshape, arange, repmat, concatenate, array, zeros
from afbf.utilities import amin, amax, matmul, floor, linspace, sign, plt
from afbf.utilities import make_axes_locatable, ndarray, pickle


class coordinates:
    """
    Handle spatial coordinates.
    """

    def __init__(self, N=-1):
        if isinstance(N, int) and N > 0:
            self.DefineUniformGrid(N)
        else:
            self.xy = None  # cartesian coordinates.
            self.ncoord = None  # number of coordinates.
            self.M = None  # grid dimension.
            self.N = None  # coordinate value corresponding to position 1.
            self.coordtype = None  # type of coordinates.

    def CheckValidity(self):
        """
        Check the validity of coordinates.

        Returns
        -------
        valid, boolean
            True if valid.

        """
        valid = isinstance(self.xy, ndarray) and self.xy.shape[1] == 2
        if valid is not True:
            print("Coordinates are not properly defined.")

        return(valid)

    def DefineUniformGrid(self, N, step=1, signed=False):
        """
        Define a uniform grid [[1,N]]x[[1,N]].
        The grid is defined in Cartesian coordinates.

        Parameters
        ----------
        N : int
            The size of the grid.
        step : int, optional
            step between grid points. The default is 1.
        signed : boolean, optional
            if True the grid is defined as  [[1,N]]x[[-N,N]].
            The default is False.

        Returns
        -------
        self.xy, self.M, self.N, self.ncoord

        """
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
        self.coordtype = 'uniformgrid'  # type of coordinates
        self.M = array([Nx, Ny])  # grid dimension
        self.N = Nx  # grid maximal dimension
        self.ncoord = N2  # number of coordinates

    def DefineNonUniformLocations(self, xy):
        """
        Define a set of locations given by xy.

        Parameters
        ----------
        xy : numpy.array of size (M, 2).
            Coordinates ; xy[m, :] contains the mth coordinates.

        Returns
        -------
        self.xy, self.M, self.N, self.ncoord

        """
        if isinstance(xy, ndarray) and xy.shape[1] == 2:
            self.coord = zeros(xy.shape, dtype=int)
            self.coord.xy = xy
            self.ncoord = xy.shape[0]
            self.M = -1
            self.N = amax(self.coord, axis=None)
            self.coordtype = 'nonuniform'
        else:
            print("DefineNonUniform...: provide xy as an ndarray (M, 2).")

    def ApplyAffineTransform(self, A):
        """
        Apply an affine transform to the coordinates.

        Parameters
        ----------
        A : numpy.array of shape (2, 2)
            Affine transform
        Returns
        -------
        self.xy

        """
        if self.CheckValidity() is False:
            return(0)

        if isinstance(A, ndarray) and A.shape[0] == 2 and A.shape[1] == 2:
            self.xy = matmul(self.xy, A)
            self.N = amax(self.xy, axis=None)
        else:
            print("ApplyAffine...: provide A as ndarray of size (2, 2).")

    def ProjectOnAxis(self, u, v):
        """
        Project coordinates on the axis oriented in (u, v).

        Parameters
        ----------
        u, v : int
            (u, v) is the projection axis.


        Returns
        -------
        array of size (M, 1)
            Projection coordinates.
        """
        if self.CheckValidity() is False:
            return(0)

        ind = u * self.xy[:, 0] + v * self.xy[:, 1]

        return ind.reshape((ind.size, 1))

    def DisplayPoints(self):
        """
        Display the points.
        """
        if self.CheckValidity() is False:
            return(0)

        plt.plot(self.xy[:, 0], self.xy[:, 1], 'rx')
        plt.axis('equal')


class sdata:
    """
    Handle spatial data including images. Images are defined for coordinates
    on a uniform grid.
    """

    def __init__(self, coord=None):
        """
        Parameters
        ----------
        coord : Coordinates, optional
            A set of coordinates where the data is defined.
            The default is None.

        Returns
        -------
        self.values, self.coord, self.name

        """
        self.name = 'undefined'
        if coord is not None:
            if isinstance(coord, coordinates):
                self.coord = coord
                self.values = zeros((coord.xy.shape[0], 1))
            else:
                print("data.__init__: provide coord as coordinates.")
                return(None)
        else:
            self.coord = None
            self.values = None

    def Display(self, nfig=1):
        """
        Display the spatial data (only defined for images).

        Parameters
        ----------
        nfig : int, optional
            Figure index. The default is 1.

        Returns
        -------
        None.

        """
        if self.coord.coordtype == 'uniformgrid':
            M = self.coord.M
            N = self.coord.N
            plt.figure(nfig)
            fig = plt.subplot()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(self.name)
            v = plt.imshow(self.values.reshape(M[1], M[0]),
                           origin='upper', cmap='gray')
            r = M[0] / M[1]
            crx = int(max(1, floor(5 * min(1, r))))
            cry = int(max(1, floor(5 * min(1, 1 / r))))
            if self.coord.xy is None:
                cx = linspace(0, M[0] / N, crx)
                cy = linspace(0, M[1] / N, cry)
            else:
                cx = linspace(amin(self.coord.xy[:, 0]) / N,
                              amax(self.coord.xy[:, 0]) / N, crx)
                cy = linspace(amin(self.coord.xy[:, 1]) / N,
                              amax(self.coord.xy[:, 1]) / N, cry)
            cy = cy[::-1]

            plt.xticks(linspace(0, M[0], crx), floor(cx * 1000) / 1000)
            plt.yticks(linspace(0, M[1], cry), floor(cy * 1000) / 1000)
            plt.ticklabel_format
            divider = make_axes_locatable(fig)
            cax = divider.append_axes("right", size="5%", pad=0.6)
            plt.colorbar(v, cax=cax)
            plt.show()
        else:
            print('SpatialData.Display: not available on non-uniform sites.')

    def Save(self, filename):
        """
        Save an image in a file

        Parameters
        ----------
        filename : str
            File name (without extension).
        """

        if self.coord.coordtype == 'uniformgrid':
            with open(filename + '.pickle', 'wb') as f:
                pickle.dump([self.coord.M, self.values], f)
        else:
            print('sdata.Save: only available for images.')

    def CreateImage(self, M):
        """
        Create an image.

        Parameters
        ----------
        M : numpy.array of shape (1, 2).
            Size of the image.

        Returns
        -------
        None.

        """
        self.coord = coordinates()
        self.coord.M = M
        self.coord.N = amax(M)
        self.coord.ncoord = M[0] * M[1]
        self.coord.coordtype = 'uniformgrid'

    def ComputeIncrements(self, hx, hy, order, M=-1):
        """
        Compute increments of an image.

        Parameters
        ----------
        hx, hy : int
            Horizontal and vertical lags.
        order : int
            Order of the increment.
        M : int, optional
            Dimension of the domain where to compute increments.
            The default is -1.

        Returns
        -------
        incre : sdata
            An image of increments.

        """
        if self.coord.coordtype == 'uniformgrid':
            N = self.coord.N
            if M == -1:
                M = self.coord.M
            else:
                if M[0] > self.coord.M[0] or M[1] > self.coord.M[1]:
                    print('ComputeIncrements: dimension error.')
                    return

            valincre = reshape(self.values, (M[0], M[1]))
            order = order + 1

            for j in range(order):
                if hy >= 0:
                    valincre = valincre[hy:M[0], 0:(M[1] - hx)]\
                        - valincre[0:(M[0] - hy), hx:M[1]]
                    M = valincre.shape
                else:
                    hy = - hy
                    valincre = valincre[0:(M[0] - hy), 0:(M[1] - hx)]\
                        - valincre[hy:M[0], hx:M[1]]

            incre = sdata()
            incre.CreateImage(M)
            incre.values = valincre
            incre.coord.N = N
            incre.name = 'Increments'
        else:
            print('data.ComputeIncrements: only available for images')

        return incre

    def SmoothImage(self, extend):
        """
        Smooth an image by averaging on local neighborhoods.

        Parameters
        ----------
        extend : int
            The extend of local neighborhood.

        Returns
        -------
        simage : data
            The smoothed image.

        """
        if self.coord.coordtype == 'uniformgrid':
            N = self.coord.N
            M = self.coord.M
            valimage = reshape(self.values, (M[0], M[1]))

            for j in range(extend):
                valimage = (valimage[1:M[0]-1, 1:M[1]-1] +
                            valimage[0:M[0]-2, 1:M[1]-1] +
                            valimage[1:M[0]-1, 0:M[1]-2] +
                            valimage[0:M[0]-2, 2:M[1]] +
                            valimage[2:M[0], 0:M[1]-2])/5

                M = valimage.shape

            simage = sdata()
            simage.CreateImage(M)
            simage.values = valimage
            simage.coord.N = N
            simage.name = 'Smoothed image.'
        else:
            print('data.SmoothImage: only available for images')

        return simage

    def ComputeLaplacian(self, scale=1):
        """
        Compute the discrete laplacian of an image.

        Parameters
        ----------
        scale : int, optional
            Scale at which the laplacian is computed. The default is 1.

        Returns
        -------
        laplacian : sdata
            Image laplacian.

        """
        if self.coord.coordtype == 'uniformgrid':
            dx2 = self.ComputeIncrements(scale, 0, 1)
            dy2 = self.ComputeIncrements(0, scale, 1)

            laplacian = sdata()
            M = array([dy2.coord.M[0], dx2.coord.M[1]])
            laplacian.CreateImage(M)
            laplacian.values = dx2.values.reshape(dx2.coord.M)[0:M[0], 0:M[1]]\
                + dy2.values.reshape(dy2.coord.M)[0:M[0], 0:M[1]]

            laplacian.coord.N = dx2.coord.N
            laplacian.name = 'Image Laplacian.'
        else:
            print('data.ComputeLaplacian: only available for images')

        return laplacian

    def ComputeImageSign(self):
        """
        Compute the sign of an image.

        Returns
        -------
        simage : sdata.
            The signed image.

        """

        if self.coord.coordtype == 'uniformgrid':
            simage = sdata()
            simage.CreateImage(self.coord.M)
            simage.values = sign(self.values)
            simage.coord.N = self.coord.N
            simage.name = 'Image sign.'
        else:
            print('ComputeImageSign: only available for images')

        return simage
