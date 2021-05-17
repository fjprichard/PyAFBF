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
================================
Textured images with large trend
================================

.. codeauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>

In this example, we show how to simulate textured images with large trends.

Basic AFBF have :term:`stationary<stationarity>` :term:`increments`.
This is ensured by the fact that the Hurst function :math:`\beta`
ranges in the interval :math:`(0,1)`. However, the package PyAFBF
enables the simulation of more generic fields for which :math:`\beta`
have values above 1. Increments of order 0 of such fields might not be
stationary anymore. They are :term:`intrinsic` fields of an order k,
which depends on the maximal value of the Hurst function. Such fields
may show large polynomial trends, the degree of which corresponding
to the order k.

.. note::
    Even when extended to an intrinsic random field,
    the hurst function of an AFBF
    has a minimum :math:`H`` which remains in :math:`(0,1)`.

.. seealso::
    The definition and theory of intrinsic random fields are presented
    in :cite:p:`Chiles-2012`. These fields have been used for modeling
    in the framework of AFBF
    :cite:p:`Richard-2017,Richard-2016,Richard-2015b,Richard-2015,
    Richard-2010`.
"""
import numpy as np
from afbf import tbfield

# Definition and simulation of an AFBF.
Z = tbfield('afbf-smooth')

Z.hurst.ChangeParameters(
    np.array([[1.3, 0.1]]),
    np.array([[-1.2, -1.1, 1.1,  1.2]])
    )
Z.hurst.trans = 1
Z.NormalizeModel()

Z.FindOrder()
Z.DisplayParameters()
np.random.seed(1)
z = Z.Simulate()
z.name = 'Intrinsic field.'
z.Display(1)
