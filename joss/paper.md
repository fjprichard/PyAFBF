
---
title: '**PyAFBF**: a Python library for sampling image textures from the anisotropic fractional Brownian field.'
tags:
- Python
- mathematics
- texture synthesis
- image processing
- probability
- statistics
- fractional Brownian field
authors:
- name: Frédéric J.P. Richard
  orcid: 0000-0001-5146-9894
  affiliation: 1
affiliations:
- name: Aix Marseille University, CNRS, Centrale Marseille, I2M, UMR 7373, Marseille, France.
  index: 1
date: 18 may 2021
bibliography: refs.bib
---

# Summary

The Python library **PyAFBF** is devoted to the simulation of anisotropic textures of image. These textures are sampled from a mathematical model called the anisotropic fractional Brownian field (AFBF) [@Bonami2003]; see \autoref{fig:example} for an illustration. The library offers several features. It enables to set, either manually or randomly, the simulated model so as to generate a wide variety of textures. It also include tools to compute model features (regularity, anisotropy,...) which may serve as attributes to describe generated textures. The library further offers the possibility to sample heterogeneous textures from random field models related to the AFBF.

![A patchwork of simulated textures. \label{fig:example}](patchwork.png)

# Statement of need

For the simulation of random fields, the library **PyAFBF** relies upon the turning-band method developed in [@Bierme-2015-TBM]. This method was historically designed to facilitate researchs on the anisotropic fractional Brownian field [@Bonami2003] and related models [@Benassi97; @Guyon2000; @polisano2014texture; @Peltier96; @Vu2020]. Simulations it generates have been particularly useful to set experimental protocols aiming at assessing performances of estimation methods [@Bierme08-ESAIM; @Richard-2010; @Richard-2015; @Richard-2015b; @Richard-2016; @Richard-2017]. The estimation issues tackled in these works concerned the functional parameters and features of the AFBF and related models. There are still largely open. The library **PyAFBF** is intended to support future experiments and help mathematicians to validate and compare their methods. In particular, it could be the basis for collaborative works aiming at the development of benchmarks and data challenges.

The simulation of random fields is a classical topic of spatial statistics [@lantuejoul2013geostatistical; @cressie2015statistics; @Chiles-2012]. As reviewed in [@liu2019advances], there are some R packages devoted to this topic, among which:

- [RandomFields](https://cran.r-project.org/web/packages/RandomFields/) [@Schlater-2015] that enables the simulation of stationary fields and also some non-stationary Gaussian random fields such max-stable fields,
    
- [FieldSim](https://cran.r-project.org/web/packages/FieldSim/FieldSim.pdf) [@brouste2007fractional] that allows the simulation of manifold indexed Gaussian field,
    

None of these package deals directly with the anisotropic fractional Brownian fields. The package FieldSim deals with mono- and multi- fractional Brownian fields but only in an isotropic setting. The package RandomFields offers a wide range of methods to simulate stationary and non-stationary, isotropic and anisotropic random fields. However, it only handles geometric and zonal anisotropies, which both differ from the one of an AFBF. Moreover, it is not specifically devoted to models derived from the fractional Brownian fields. Hence, the package **PyAFBF** is complementary to these R packages.

The implementation of random field simulation methods in Python are less manifest. It is the purpose of the package [python-randomfields](https://github.com/dubourg/python-randomfields) and [dune-randomfield](https://gitlab.dune-project.org/oklein/dune-randomfield). It is also a part of packages [spam](https://ttk.gricad-pages.univ-grenoble-alpes.fr/spam/index.html) and [dorie](https://hermes.iup.uni-heidelberg.de/dorie_doc/master/html/index.html). But, none of these package enables the simulation of AFBF. Hence, the package *AFBF* offers original simulation tools to a large community of Python developers. This is of interest for researchers in image processing where random fields can serve as texture or noise models for medical images [@Bierme09-ESAIM; @Bierme10-Springer; @Richard-2010; @Richard-2015b; @Richard-2016] or photographic films [@Richard-2017]. This is also of interest for researchers in machine learning where the random field simulation could be included in image generative models such as GAN.

# Definition and simulation of an AFBF.

An AFBF $Z$ is a Gaussian non-stationary random field with stationary increments whose semi-variograms are of a form

$$
v(h) = \frac{1}{2} \mathbb{E}((Z(x+h) - Z(x))^2) = \frac{1}{2} \int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} \tau(\theta) \left\vert h, u(\theta) \rangle \right\vert^{2\beta(\theta)} d\theta, \: u(\theta) = (\cos \theta, \sin \theta),
$$

which is characterized by two $\pi$-periodic functions $\tau$ and $\beta$ called the topothesy function and the Hurst function, respectively. These functions determined the properties of the AFBF and the aspect of the textures that are sampled from it.

The package **PyAFBF** proposes some convenient representations for these functions (Fourier, step functions,...) that enable to easily set an AFBF, either manually or at random, in a proper way.

Using the package **PyAFBF**, image textures are realizations of an AFBF on a discrete grid. This are simulated using a turning band fields. These fields are defined, for some set of angles $(\varphi_k, k=1,\cdots,K)$ in $[-\frac{\pi}{2}, \frac{\pi}{2}]$ and of appropriate non-negative weights $(\lambda_k, k=1,\cdots,K)$, as

$$ Z_{\varphi} (x)  = \sum_{k=1}^K \lambda_k X_k ( \langle u(\varphi_k), x \rangle ), $$

where $X_k$ are independent Brownian motions with Hurst index $h_k$. The package includes a Python class to handle turning-band fields, simulate them and compute their properties.

# Availability and Community Guidelines

The package **PyAFBF** can be downloaded from the Github [repository](https://github.com/fjprichard/PyAFBF). A documentation, which includes a quickstart guide, a gallery of examples and API, is available at the [PyAFBF](https://fjprichard.github.io/PyAFBF/) site. Users and contributors are welcome to contribute, request features, and report bugs via Github.

# References
