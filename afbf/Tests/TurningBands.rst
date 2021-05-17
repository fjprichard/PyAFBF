Tests for the module Turning Bands
==================================


Compare the semi-variogram and its turning-band approximate
-----------------------------------------------------------

>>> from numpy import mean, absolute, array, pi, diff, arange
>>> from afbf import coordinates, perfunction, tbfield
>>> N = 512

>>> lags = coordinates()
>>> lags.DefineUniformGrid(50, 1, True)
>>> lags.N = N


>>> Z = tbfield()
>>> Z.hurst.ChangeParameters(array([0.5]))
>>> Z.NormalizeModel()
>>> Z.ComputeSemiVariogram(lags)
>>> Z.ComputeApproximateSemiVariogram(lags)
>>> mean(absolute(Z.vario.values - Z.svario.values)) < 10e-4
True

>>> hurst = perfunction('step-smooth', 2)
>>> topo = perfunction('step-smooth', 2)
>>> hurst.ChangeParameters(array([0.6, 0.2]), array([-pi/4, -pi/8, pi/8, pi/4]))
>>> Z = tbfield('field', topo, hurst)
>>> Z.NormalizeModel()
>>> Z.ComputeSemiVariogram(lags)
>>> Z.ComputeApproximateSemiVariogram(lags)
>>> mean(absolute(Z.vario.values - Z.svario.values)) < 10e-4
True

>>> hurst = perfunction('step', 2)
>>> topo = perfunction('step', 2)
>>> hurst.ChangeParameters(array([0.6, 0.2]), array([-pi/8, pi/8]))
>>> Z = tbfield('field', topo, hurst)
>>> Z.NormalizeModel()
>>> Z.ComputeSemiVariogram(lags)
>>> Z.ComputeApproximateSemiVariogram(lags)
>>> mean(absolute(Z.vario.values - Z.svario.values)) < 10e-4
True

Compare the field variogram and its estimate from the simulated field
---------------------------------------------------------------------

>>> coord = coordinates(N)
>>> z = Z.Simulate(coord)
>>> evario = z.ComputeEmpiricalSemiVariogram(lags)
>>> mean(absolute(Z.svario.values - evario.values)) < 10e-3
True

>>> Z = tbfield()
>>> Z.hurst.ChangeParameters(array([0.5]))
>>> Z.NormalizeModel()
>>> Z.ComputeApproximateSemiVariogram(lags)
>>> coord = coordinates(N)
>>> z = Z.Simulate(coord)
>>> evario = z.ComputeEmpiricalSemiVariogram(lags)
>>> mean(absolute(Z.svario.values - evario.values)) < 10e-3
True

Turning band computation
------------------------

>>> from afbf.Simulation.TurningBands import tbparameters
>>> from numpy import tan

>>> tb = tbparameters(500)
>>> mean(diff(tb.Kangle)) < 10e-3
True

>>> mean(absolute(tan(tb.Kangle[1:-1]) - tb.Pangle[1:-1] / tb.Qangle[1:-1])) < 10e-15
True

>>> mean(tb.Kangle[0:328] + tb.Kangle[arange(656,328, -1)]) < 10e-6
True