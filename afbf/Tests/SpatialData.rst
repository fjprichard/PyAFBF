Tests for the module SpatialData
================================


Class coordinates
-----------------

Creation of a uniform grid

>>> from afbf import coordinates
>>> grid = coordinates(5)
>>> grid.xy
array([[1, 5],
       [2, 5],
       [3, 5],
       [4, 5],
       [5, 5],
       [1, 4],
       [2, 4],
       [3, 4],
       [4, 4],
       [5, 4],
       [1, 3],
       [2, 3],
       [3, 3],
       [4, 3],
       [5, 3],
       [1, 2],
       [2, 2],
       [3, 2],
       [4, 2],
       [5, 2],
       [1, 1],
       [2, 1],
       [3, 1],
       [4, 1],
       [5, 1]])

>>> grid.Nx
5

>>> grid.Ny
5

>>> grid.N
5

Application of an affine transform.

>>> from numpy import array
>>> grid = coordinates(5)
>>> A = array([[1, 2], [4, 3]])
>>> grid.ApplyAffineTransform(A)
>>> grid.xy
array([[21, 17],
       [22, 19],
       [23, 21],
       [24, 23],
       [25, 25],
       [17, 14],
       [18, 16],
       [19, 18],
       [20, 20],
       [21, 22],
       [13, 11],
       [14, 13],
       [15, 15],
       [16, 17],
       [17, 19],
       [ 9,  8],
       [10, 10],
       [11, 12],
       [12, 14],
       [13, 16],
       [ 5,  5],
       [ 6,  7],
       [ 7,  9],
       [ 8, 11],
       [ 9, 13]])

Projection on an axis.

>>> grid = coordinates(5)
>>> grid.ProjectOnAxis(2, 3)
array([[17],
       [19],
       [21],
       [23],
       [25],
       [14],
       [16],
       [18],
       [20],
       [22],
       [11],
       [13],
       [15],
       [17],
       [19],
       [ 8],
       [10],
       [12],
       [14],
       [16],
       [ 5],
       [ 7],
       [ 9],
       [11],
       [13]])


Class sdata
-----------

>>> from numpy import mean, power, absolute, nonzero
>>> from afbf import coordinates, sdata

>>> grid = coordinates()
>>> grid.DefineUniformGrid(256, step=1, signed=True)
>>> image = sdata(grid)

Computation of increments
-------------------------

>>> image.values[:, 0] = grid.xy[:, 0]

>>> increm = image.ComputeIncrements(1, 0)
>>> nonzero(increm.values != 1)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(1, 0, 1)
>>> nonzero(increm.values != 0)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(-1, 0)
>>> nonzero(increm.values != -1)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(2, 0)
>>> nonzero(increm.values != 2)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(0, 1)
>>> nonzero(increm.values != 0)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(0, -1)
>>> nonzero(increm.values != 0)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(1, 1)
>>> nonzero(increm.values != 1)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(-1, -1)
>>> nonzero(increm.values != -1)
(array([], dtype=int64), array([], dtype=int64))

>>> image.values[:, 0] = grid.xy[:, 1]
>>> increm = image.ComputeIncrements(1, 0)
>>> nonzero(increm.values != 0)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(-1, 0)
>>> nonzero(increm.values != 0)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(0, 1)
>>> nonzero(increm.values != 1)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(0, -1)
>>> nonzero(increm.values != -1)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(0, 2)
>>> nonzero(increm.values != 2)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(1, 1)
>>> nonzero(increm.values != 1)
(array([], dtype=int64), array([], dtype=int64))

>>> increm = image.ComputeIncrements(-1, -1)
>>> nonzero(increm.values != -1)
(array([], dtype=int64), array([], dtype=int64))


Computation of quadratic variations
-----------------------------------

>>> image.values[:, 0] = grid.xy[:, 0]
>>> lags = coordinates()
>>> lags.DefineUniformGrid(50, 1, True)
>>> varq = image.ComputeQuadraticVariations(lags)
>>> nonzero(varq.values[:, 0] - power(lags.xy[:, 0], 2) != 0)
(array([], dtype=int64),)

>>> image.values[:, 0] = grid.xy[:, 1]
>>> varq = image.ComputeQuadraticVariations(lags)
>>> nonzero(varq.values[:, 0] - power(lags.xy[:, 1], 2) != 0)
(array([], dtype=int64),)


Computation of the laplacian
----------------------------

>>> image.values[:, 0] = power(grid.xy[:, 0], 2) + power(grid.xy[:, 1], 2) 
>>> lap = image.ComputeLaplacian(1)
>>> nonzero(lap.values[:, 0]  != lap.values[0, 0])
(array([], dtype=int64),)
