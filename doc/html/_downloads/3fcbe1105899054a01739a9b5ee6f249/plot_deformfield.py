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
============================================
Anisotropic textures from field deformations
============================================

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>

In this example, we show how to simulate
:term:`anisotropic<isotropy>` textures from the deformation
of an :term:`isotropic<isotropy>` random field.

Given a fractional Brownian field :math:`X` and  an affine transformation
:math:`T`, the transformed field is defined by composition

.. math::
    Y = X \circ T.

The anisotropic texture is obtained by simulation of :math:`Y`.

.. note::
    This approach could also be applied to more generic afbf.

.. seealso::
   Deformed fields have been of interest in several studies;
   see for instance :cite:p:`Guyon2000`.
"""
import numpy as np
from afbf import coordinates, tbfield

# Define a fractional Brownian field.
X = tbfield()
# Define an affine transform.
T = np.array([[1, 0], [2, 1]], dtype=int)

# Define a uniform grid.
coord = coordinates(256)
# Simulate the field without transformation.
n0 = int(np.random.randn())
np.random.seed(n0)
y0 = X.Simulate()

# Apply the coordinate transform to the grid.
coord.ApplyAffineTransform(T)
# Simulate the field with transformation (with a same seed).
np.random.seed(n0)
y = X.Simulate(coord)

# Display of simulations.
y.name = 'Simulation of the deformed field.'
y.Display(1)

y0.name = 'Simulation of the undeformed field.'
y0.Display(2)
