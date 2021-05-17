Tests for the module RandomProcess
==================================

>>> from numpy import linspace, zeros, log, mean, absolute, arange, power
>>> from afbf import process
>>> model = process('fbm', param=0.2)
>>> T = 10

>>> model.ComputeFBMAutocovariance(T)
>>> model.autocov
array([[ 1.        ],
       [-0.34024604],
       [-0.04358512],
       [-0.02154106],
       [-0.01335137],
       [-0.00926712],
       [-0.00689233],
       [-0.00537181],
       [-0.00433116],
       [-0.00358311]])

>>> model.ComputeAutocovarianceSpectrum()
>>> model.spect
array([[0.10724486+0.00000000e+00j],
       [0.29765627+1.38777878e-17j],
       [0.52243366+0.00000000e+00j],
       [0.73926324+5.55111512e-17j],
       [0.96253527+0.00000000e+00j],
       [1.1711565 +5.55111512e-17j],
       [1.35570274+0.00000000e+00j],
       [1.49799039-6.93889390e-18j],
       [1.58958187+4.80740672e-17j],
       [1.62011522+1.16466626e-16j],
       [1.58958187-1.80277752e-17j],
       [1.49799039-5.17926092e-17j],
       [1.35570274+0.00000000e+00j],
       [1.1711565 -6.93889390e-18j],
       [0.96253527-4.80740672e-17j],
       [0.73926324-1.71977777e-16j],
       [0.52243366+1.80277752e-17j],
       [0.29765627-3.71854204e-18j]])

Semi-variogram
--------------

>>> model.ComputeFBMSemiVariogram(linspace(0, 1, T))
>>> model.vario
array([0.        , 0.20762182, 0.27395864, 0.32219701, 0.36149059,
       0.3952401 , 0.4251415 , 0.4521809 , 0.47698969, 0.5       ])

Simulation : validation by parameter estimation
-----------------------------------------------

>>> N = 10000
>>> T = 5
>>> H = 0.2
>>> model = process('fbm', param=H)
>>> t = arange(1, T+1, dtype=int)
>>> model.ComputeFBMSemiVariogram(t)
>>> model.Simulate(N)
>>> v = zeros(t.size)
>>> for j in range(t.size): v[j] = 0.5 * mean(power(model.y[0:-t[j], 0] - model.y[t[j]:, 0], 2))
>>> mean(absolute(0.5 * log(v[1:] / v[0:-1]) / log(t[1:] / t[0:-1]) - H)) < 10e-2
True
>>> absolute(model.vario[0] - v[0]) < 10e-4
True
