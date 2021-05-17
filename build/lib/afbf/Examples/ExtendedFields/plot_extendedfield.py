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
r"""
========================================
Textured images with spectral variations
========================================

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>

In this example, we show how to simulate a field having a topothesy whose
values may vary depending both on the direction and the amplitude.


.. note::
    The obtained simulation relates to a model with a semi-variogram
    of the form:

    .. math::
        v(x) = \int_{\mathbb{R}^2} \vert e^{i\langle x, w \rangle} - 1 \vert^2
        \tau(w) \vert w \vert^{2H-2} dw.

    where, contrarily to usual AFBF, the value :math:`\tau(w)` does not only
    depend on the direction :math:`\arg(w)` of :math:`w`,
    but also on its amplitude :math:`\vert w \vert`.
"""
import numpy as np
from afbf import perfunction, tbfield
from matplotlib import pyplot as plt

nb_angle = 4  # Number of angle subdivisions.
nb_spect = 3  # Number of spectrum subdivisions.

# Definition of a fractional Brownian field.
Z = tbfield('fbf')
Z.hurst.ChangeParameters(
    np.array([0.1])
    )
Z.NormalizeModel()


Z.ExtendTopothesy(nb_spect, nb_angle)
for j in range(nb_spect):
    # Set uniformly spaced interval for the step functions.
    Z.gtopo[j].SetUniformStepInterval()

np.random.seed(1)
z = Z.Simulate()
z.name = 'Extended field.'
z.Display(1)


# Visualize the generalized topothesy.
gtopo = perfunction('step-smooth', nb_spect)  # Topothesy at an angle.
gtopo.SetUniformStepInterval()

z = np.linspace(-np.pi / 2, np.pi / 2, 1000)
im = np.zeros((z.size, z.size))
for i in range(z.size):
    # Setting parameters of the topothesy at angle z[i]
    for j in range(nb_spect):
        Z.gtopo[j].Evaluate(np.array([z[i]]))
        # The value of Z.gtopo[j] at z[i] gives the values of the jth
        # parameter of the generalized topothesy at the ith angle.
        gtopo.fparam[0, j] = Z.gtopo[j].values[0, 0]

    # Evaluate the topothesy at angle z[i]
    gtopo.Evaluate(z)
    im[i, :] = gtopo.values.reshape((1, z.size))

plt.figure(2)
plt.imshow(im, cmap='gray')
plt.title('Generalized topothesy')
plt.xlabel(r'Amplitude $\rho$')
plt.ylabel(r'Angle $\varphi$')
loc, lab = plt.yticks(
    np.linspace(0, z.size, 5),
    [r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$']
    )
