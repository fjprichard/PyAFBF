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
=========================
Fractional Brownian field
=========================

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>

This example shows how to simulate a fractional Brownian field with
a prescribed Hurst index.

.. note::

"""

import numpy as np
from afbf import tbfield

N = 256  # Image size.
H = 0.2  # Hurst index in (0, 1).

# Define the field.
Z = tbfield('fbf')

# Change the parameter of the Hurst function.
Z.hurst.ChangeParameters(np.array([[H]]))
Z.NormalizeModel()

# Simulate the field.
z = Z.Simulate()

# Display the simulation.
z.Display()
