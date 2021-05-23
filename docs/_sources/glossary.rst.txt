Glossary
========

.. sectionauthor:: Frédéric Richard <frederic.richard_at_univ-amu.fr>

.. glossary:: 
   random field

A random field :math:`Z` is a collection of random variables :math:`Z(x)` which are located at positions :math:`x` of a multidimensional space (for instance, the plane :math:`\mathbb{R}^2`). A collection of variables defined on :math:`\mathbb{R}` is rather called a random process. 

The random field is Gaussian if any linear combination :math:`\sum_{i=1}^n \lambda_i Z(x_i)` is a Gaussian random variable. 
The probability distribution of a Gaussian field is characterized by 

- its expectation function:
	 .. math:: x \rightarrow \mathbb{E}(Z(x)), 

- its covariance function:
	 .. math:: (x, y) \rightarrow \mathrm{cov}(Z(x), Z(y)). 

.. glossary::
   stationarity

A :term:`random field` :math:`Z` is (second-order) stationary if :math:`\mathbb{E}(Z(x))` does not depend on the position :math:`x` (is constant) and if :math:`\mathrm{cov}(Z(x), Z(y))` only depends on the relative position :math:`x-y`. First and second-order properties of such a field are the same all over the space. 

.. glossary::
   increments

Increments :math:`W` (of order 0) of a :term:`random field` :math:`Z` are random variables of the form

.. math:: W = \sum_{i=1}^n \lambda_i Z(x_i),

where :math:`\lambda_i` are such that  :math:`\sum_{i=1}^n \lambda_i =0`. For instance,  :math:`Z(x+h) - Z(x)` is an increment of :math:`Z`. More generally, increments of order :math:`k \in \mathbb{N}` are increments such that :math:`\sum_{i=1}^n \lambda_i P(x_i) =0` for any polynomial :math:`P` of order :math:`k`.  

An increment field :math:`W` is a set of increments :math:`W(y)` defined at any position :math:`y` by

.. math:: W(y) = \sum_{i=1}^n \lambda_i Z(y + x_i).

.. glossary::
   intrinsic

An intrinsic :term:`random field` of order :math:`k` is a :term:`random field` whose :term:`increments` of order :math:`k` 
are :term:`stationary<stationarity>`  :cite:p:`Chiles-2012,Richard-2017,Richard-2016,Richard-2015b,Richard-2015,Richard-2010`. An intrinsic field of order 0 is simply called random fields with stationary increments.

.. glossary::
   semi-variogram

Let :math:`Z` be a :term:`random field` with :term:`stationary<stationarity>` :term:`increments`. The semi-variogram of :math:`Z` 
is defined, for any :math:`h`, by

.. math:: v(h) = \frac{1}{2} \mathbb{E}((Z(h) - Z(0))^2) = \frac{1}{2} \mathbb{E}((Z(x+h) - Z(x))^2), \forall x.

.. glossary::
   density

Let :math:`Z` be a :term:`random field` with :term:`stationary<stationarity>` :term:`increments`. A non-negative and even function :math:`f` is the density of :math:`Z`
if

.. math:: v(h) = \int_{\mathbb{R}^2} \vert e^{i\langle w, h\rangle} -1 \vert^2 f(w) dw,
 
The density of an AFBF is of the form

.. math::    f(w)=\tau(\arg(w)) |w|^{-2\beta(\arg(w))-2}, w \in \mathbb{R}^2,

where :math:`\tau` and :math:`\beta` are non-negative :math:`\pi`-periodic
functions depending both on the direction :math:`\arg(w)` of
the frequency :math:`w`.

.. glossary::
   regularity

The regularity (in the Hölder sense) of a random field :math:`Z` is the highest value :math:`H \in (0, 1)` for which

.. math:: 
    \vert Z(y) - Z(x) \vert \leq c \vert y - x \vert^\alpha   

holds with probability 1 for any :math:`\alpha < H` and :math:`x, y` in any arbitrary compact set.

.. image:: ./Figures/regularity2.png

.. image:: ./Figures/regularity8.png

.. glossary::
   isotropy

A field is isotropic if its properties are the same in all space directions. A Gaussian :term:`random field` with :term:`stationary<stationarity>` :term:`increments` and :term:`density` :math:`f` is isotropic if and only if :math:`f` is radial, ie.

   .. math:: f(w) = \tilde{f}(\vert w \vert), \forall w,

meaning that values of :math:`f` does not depend on the direction :math:`\arg(w)` of :math:`w`, but only on its module :math:`\vert w \vert`. 

A field is anisotropic if it is not :term:`isotropic<isotropy>`.

The difference between realizations of isotropic and anisotropic fields is illustrated below.

.. image:: ./Figures/isotropic.png

.. image:: ./Figures/anisotropic.png