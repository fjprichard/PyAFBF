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
=================
Texture Patchwork
=================

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>

This example shows how to build a series of simulations of a field model by
changing its parameter values at random. This is useful to construct a dataset
of textures.
"""
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


import numpy as np

from afbf import tbfield, coordinates

ncols = 10
nrows = 10
nbexamples = ncols * nrows  # Number of examples.
N = 100  # Size of examples.

# Mode of simulation for step values (alt, 'unif', 'unifmax', or 'unifrange').
simstep = 'unifmin'
# Mode of simulation for step interval bounds (alt, 'nonunif').
simbounds = 'unif'

# Define the field to be simulated and coordinates where to simulate.
field = tbfield('afbf-smooth')
coord = coordinates(N)

# Set the mode of simulation for the Hurst function.
field.hurst.SetStepSampleMode(mode_cst=simstep, mode_int=simbounds)

# Prepare the figure
fig = plt.figure(figsize=(nrows, ncols))
gs = gridspec.GridSpec(nrows, ncols)
gs.update(wspace=0.05, hspace=0.05)  # Set axe spacing.

# Generate several examples.
for example in range(nbexamples):
    # Sample new model parameters.
    np.random.seed(example)
    field.hurst.ChangeParameters()
    field.topo.ChangeParameters()
    # Uncomment to show field parameters.
    # field.DisplayParameters()

    # Compute field features.
    field.ComputeFeatures()
    # Uncomment to show some field features.
    # print('Hurst index:', field.H)
    # print('Std deviation (hurst):', field.aniso_indices_hurst[0])
    # print('TV-norm (hurst):', field.aniso_sharpness_hurst[0])

    # Simulate an example with the current model.
    np.random.seed(example)
    field.EvaluateTurningBandParameters()
    simu = field.Simulate(coord)
    # Uncomment to display the field simulation.
    # simu.Display(2)

    # To handle simu as an ndarray of numpy, set
    image = np.reshape(simu.values, simu.M)
    # To further display it with pyplot of matplotlib:
    i = int(np.floor(example / ncols))
    j = int(example - i * ncols)
    ax = plt.subplot(gs[i, j])
    ax.imshow(image, cmap='gray')
    ax.set_axis_off()

plt.show()
