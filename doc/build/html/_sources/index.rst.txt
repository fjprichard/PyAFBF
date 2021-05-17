**********************************
Welcome to PyAFBF's documentation!
**********************************

The Package PyAFBF is intended for the simulation of rough anisotropic image textures. Textures are generated from a mathematical model called the anisotropic fractional Brownian field. Some texture examples are shown below on the patchwork. 

 .. image:: ./Figures/patchwork.png

.. note::

	The anisotropic fractional Brownian field was introduced in :cite:p:`Bonami2003` and further studied in several works
	:cite:p:`Bierme08-ESAIM,Richard-2017,Richard-2016,Richard-2015b,Richard-2015,Richard-2010,Vu2020`. It was applied for the modeling 	and the analysis of medical images :cite:p:`Bierme10-Springer,Bierme09-ESAIM,Richard-2010,Richard-2015b,Richard-2016` or 		photographic films :cite:p:`Richard-2017` . The simulation method was designed in :cite:`Bierme-2015-TBM`.

Package features
================

- Simulation of rough anisotropic textures,

- Computation of field features (semi-variogram, regularity, anisotropy indices),

- Random definition of simulated fields,

- Extensions to other fields (deformed fields, intrinsic fields, heterogeneous fields, binary patterns).


Installation from sources
=========================

In the root directory of the package, just do

.. code-block:: python

    python setup.py install
    or
    pip install -e .

Communication to the author
===========================

PyAFBF is developed and maintained by Frédéric Richard. For feed-back, contributions, bug reports, contact directly the author at
<frederic.richard_at_univ-amu.fr>.

Citation
========

When using PyAFBF, please cite the original paper :cite:`Bierme-2015-TBM`. 

Licence
=======

PyAFBF is under licence GNU GPL, version 3.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ./quickstartguide.rst
   ./auto_examples/index.rst
   ./afbf.Field.rst
   ./afbf.Classes.rst
   ./glossary.rst
   ./refs.rst

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

