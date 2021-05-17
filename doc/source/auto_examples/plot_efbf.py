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
====================================
Elementary Fractional Brownian field
====================================

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>

This example shows how to simulate an elementary fractional Brownian field
with prescribed Hurst index, step intervals and directions.

.. note::

"""
import numpy as np
from afbf import tbfield

N = 256  # Image size.
H = 0.2  # Hurst index in (0, 1).
T = np.pi / 8  # Interval bound for selected frequencies.
D = np.pi / 3  # Direction.

# Define the field.
Z = tbfield('efbf')

# Change the parameter of the Hurst and topothesy functions.
Z.hurst.ChangeParameters(fparam=np.array([[H]]))
Z.topo.ChangeParameters(fparam=np.array([0, 1]), finter=np.array([-T, T]))
# Translate the topothesy function to be at the right orientation.
Z.topo.ApplyTransforms(translate=-D)

# Simulate the field.
z = Z.Simulate()

# Display the simulation.
z.Display()
