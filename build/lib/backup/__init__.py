#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Frédéric Richard, AMU."
__copyright__ = "Copyright, 2021."
__credits__ = ["Frédéric Richard"]
__license__ = "GNU GENERAL PUBLIC LICENSE, Version 3."
__version__ = "1.0.0"
__maintainer__ = "Frédéric Richard"
__email__ = "frederic.richard_at_univ-amu.fr"
__status__ = "Release"

"""
Import main classes of the package
"""

# Classes for coordinates and spatial data.
from afbf.Classes.SpatialData import coordinates, sdata

# Class for periodic functions.
from afbf.Classes.PeriodicFunction import perfunction

# Class for random processes (1D).
from afbf.Classes.RandomProcess import process

# Class for fields.
from afbf.Classes.Field import field

# Class for fields.
from afbf.Simulation.TurningBands import tbfield
