.. image:: https://zenodo.org/badge/368267301.svg
   :target: https://zenodo.org/badge/latestdoi/368267301

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

The package can be installed through PYPI with
 
 pip install PyAFBF
 
To install the package in a Google Collab environment, please type

 !pip install imgaug==0.2.6
 
 !pip install PyAFBF

Communication to the author
===========================

PyAFBF is developed and maintained by Frédéric Richard. For feed-back, contributions, bug reports, contact directly the `author <https://github.com/fjprichard>`_, or use the `discussion <https://github.com/fjprichard/PyAFBF/discussions>`_ facility.


Licence
=======

PyAFBF is under licence GNU GPL, version 3.

Contents
========

    - Quick start guide
       - Getting started
       - Customed models
       - Tuning model parameters
       - Model features
       - Simulating with turning-band fields
    - Example gallery
       - Basic examples
       - Extended anisotropic fields
       - Heterogeneous fields
       - Related anisotropic fields
    - API: main classes
       - AFBF (field)
       - Turning band field (tbfield)
    - API: auxiliary classes
       - Periodic functions (perfunction)
       - Coordinates (coordinates)
       - Spatial data (sdata)
       - Process (process)
       - Turning bands (tbparameters)
       - ndarray

