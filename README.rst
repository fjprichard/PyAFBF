The Package PyAFBF is intended for the simulation of rough anisotropic image textures. Textures are sampled from a mathematical model called the anisotropic fractional Brownian field. More details can be found on the `documentation <https://fjprichard.github.io/PyAFBF/>`_.

Package features
================

- Simulation of rough anisotropic textures,

- Computation of field features (semi-variogram, regularity, anisotropy indices) that can serve as texture attributes,

- Random definition of simulated fields,

- Extensions to related fields (deformed fields, intrinsic fields, heterogeneous fields, binary patterns).


Installation from sources
=========================

The package source can be downloaded from the `repository <https://github.com/fjprichard/PyAFBF>`_. 

To install the package, write the following commands in the root directory where the package was downloaded:

    python setup.py install
    or
    pip install -e .

After installation, the package can be tested by running 

    python tests/TestAll.py

If no output appears, then the package is correctly installed.

Communication to the author
===========================

PyAFBF is developed and maintained by Frédéric Richard. For feed-back, contributions, bug reports, contact directly the `author <https://github.com/fjprichard>`_, or use the `discussion <https://github.com/fjprichard/PyAFBF/discussions>`_ facility.


Licence
=======

PyAFBF is under licence GNU GPL, version 3.
