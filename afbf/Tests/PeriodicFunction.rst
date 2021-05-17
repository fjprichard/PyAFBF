Tests for the module PeriodicFunction
=====================================

>>> from afbf import perfunction
>>> from numpy import array, linspace, pi

>>> t = linspace(-pi, pi, 12, endpoint=True)

>>> f = perfunction('step-ridge', 1)
>>> f.fparam.shape
(1, 2)
>>> f.finter.shape
(1, 2)
>>> f.ChangeParameters(array([0, 1]), array([-pi / 4, pi /4]))
>>> f.Evaluate(t)
>>> f.values
array([[1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1.]])

>>> f = perfunction('step-constant')
>>> f.fparam.shape
(1, 1)
>>> f.finter.shape
(1, 1)
>>> f.ChangeParameters(array([1]))
>>> f.Evaluate(t)
>>> f.values
array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

>>> f = perfunction('step-smooth', 2)
>>> f.fparam.shape
(1, 2)
>>> f.finter.shape
(1, 4)
>>> f.ChangeParameters(array([0.2, 0.5]), linspace(- 3 * pi / 8, 3 * pi / 8, 4, endpoint=True))
>>> f.trans = 0
>>> f.Evaluate(t)
>>> f.values
array([[0.35      , 0.5       , 0.5       , 0.26890388, 0.2       ,
        0.2135552 , 0.4864448 , 0.5       , 0.43109612, 0.2       ,
        0.2       , 0.35      ]])

>>> f = perfunction('step', 3)
>>> f.fparam.shape
(1, 3)
>>> f.finter.shape
(1, 3)
>>> f.ChangeParameters(array([0.2, 0.5, 0.1]), linspace(- 3 * pi / 8, 3 * pi / 8, 3, endpoint=True))
>>> f.Evaluate(t)
>>> f.values
array([[0.1, 0.1, 0.1, 0.2, 0.5, 0.5, 0.1, 0.1, 0.2, 0.5, 0.5, 0.1]])

>>> f = perfunction('Fourier', 2)
>>> f.fparam.shape
(1, 5)
>>> f.ChangeParameters(array([1, 0.1, 0.2, 0.4, 0.3]))
>>> f.Evaluate(t)
>>> f.values
array([[1.3       , 1.50114702, 0.91140436, 1.12180063, 0.48246111,
        0.67806243, 1.65635428, 1.10527873, 1.02280219, 0.90069756,
        0.31999168, 1.3       ]])

>>> f.ApplyTransforms(-0.56)
>>> f.Evaluate(t)
>>> f.values
array([[1.51483088, 0.91227735, 1.12302648, 0.49714274, 0.65581182,
        1.65216251, 1.11931847, 1.01587491, 0.91547093, 0.3167017 ,
        1.27738221, 1.51483088]])
