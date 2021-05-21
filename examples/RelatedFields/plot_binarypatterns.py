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
===============
Binary textures
===============

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>

In this example, we show how to simulate
texture images with binary patterns.

Binary patterns are obtained by applying a Laplacian operator to simulated
fields, which highlights the field geometry.

"""
import numpy as np
from afbf import perfunction, tbfield

# Define an AFBF.
topo = perfunction('Fourier', 3, 'Topo')
hurst = perfunction('step', 2, 'Hurst')
Z = tbfield('afbf', topo, hurst)
Z.hurst.ChangeParameters(np.array([[0.5, 0.1]]), np.array([[-1.18, 1.3]]))
Z.topo.ChangeParameters(np.array([[2, 1, 0, 0.06, -0.15, 0.64, 0.25]]))

# Simulate the field.
np.random.seed(1)
z = Z.Simulate()

# Compute the Laplacian of the simulation at scale 20.
laplacian = z.ComputeLaplacian(15)
# Compute its sign.
patterns = laplacian.ComputeImageSign()

# Display.
patterns.name = 'Binary patterns'
patterns.Display()
