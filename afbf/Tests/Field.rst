Tests for the module Field
==========================


Test the predefined fields
--------------------------

>>> from numpy import array, mean, absolute, power
>>> from afbf import field, perfunction, coordinates

>>> Z = field()
>>> Z.topo.ftype == 'step-constant'
True
>>> Z.topo.fparam.shape
(1, 1)
>>> Z.topo.finter.shape
(1, 1)
>>> Z.hurst.ftype == 'step-constant'
True
>>> Z.hurst.fparam.shape
(1, 1)
>>> Z.hurst.finter.shape
(1, 1)

>>> Z = field('efbf')
>>> Z.topo.ftype == 'step-ridge'
True
>>> Z.topo.fparam.shape
(1, 2)
>>> Z.topo.finter.shape
(1, 2)
>>> Z.hurst.ftype == 'step-constant'
True
>>> Z.hurst.fparam.shape
(1, 1)
>>> Z.hurst.finter.shape
(1, 1)

>>> Z = field('afbf')
>>> Z.hurst.fparam.shape
(1, 2)
>>> Z.hurst.finter.shape
(1, 2)
>>> (Z.hurst.finter[0, 0] == Z.topo.finter[0, 0]) and (Z.hurst.finter[0, 1] == Z.topo.finter[0, 1])
True

>>> Z = field('afbf-smooth')
>>> (Z.topo.ftype == Z.hurst.ftype) and (Z.hurst.ftype == 'step-smooth')
True

>>> Z = field('afbf-Fourier')
>>> (Z.topo.ftype == 'Fourier') and (Z.hurst.ftype == 'step')
True

>>> Z = field('afbf-smooth-Fourier')
>>> (Z.topo.ftype == 'Fourier') and (Z.hurst.ftype == 'step-smooth')
True

Test the normalization
----------------------

>>> Z = field()
>>> Z.hurst.ChangeParameters(array([0.5]))
>>> Z.NormalizeModel()
>>> Z.topo.fparam[0, 0]
0.5

Test the semi-variogram
-----------------------

>>> lags = coordinates()
>>> lags.DefineUniformGrid(50, 1, True) 
>>> Z.ComputeSemiVariogram(lags)
>>> v = 0.5 * power(power(lags.xy[:, 0], 2) +  power(lags.xy[:, 1], 2), 0.5) / power(lags.N, 2 * 0.5)
>>> mean(absolute(v - Z.vario.values.reshape(v.shape))) < 1e-16
True



