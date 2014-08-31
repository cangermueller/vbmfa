=======================================================
vbmfa: Variational Bayesian Mixture of Factor Analysers
=======================================================

Variational Bayesian Mixture of Factor Analysers for dimensionality reduction
and clustering.

Factor analysis (FA) is a method for dimensionality reduction, similar to
principle component analysis (PCA), singular value decomposition (SVD), or
independent component analysis (ICA). Applications include visualization, image
compression, or feature learning. A mixture of factor analysers consists of
several factor analysers, and allows both dimensionality reduction and
clustering. Variational Bayesian learning of model parameters prevents
overfitting compared with maximum likelihood methods such as expectation
maximization (EM), and allows to learn the dimensionality of the lower
dimensional subspace by automatic relevance determination (ARD). A detailed
explanation of the model can be found `here
<https://github.com/cangermueller/vbmfa/raw/master/docs/data/model.pdf>`_.

.. note::

  The current version is still under development, and needs to be optimized for
  large-scale data set. I am open for any suggestions, and happy about every
  bug report!

Installation
------------
The easiest way to install vbmfa is to use PyPI::

  pip install vbmfa

Alternatively, you can checkout the repository from Github::

  git clone https://github.com/cangermueller/vbmfa.git

Examples
--------
The folder ``examples/`` contains example ipython notebooks:

- `VbFa <http://nbviewer.ipython.org/github/cangermueller/vbmfa/blob/master/
  examples/140709_vbfa.ipynb>`_, a single Variational Bayesian Factor Analyser
- `VbMfa <http://nbviewer.ipython.org/github/cangermueller/vbmfa/blob/master/
  examples/140709_vbmfa.ipynb>`_, a mixture of Variational Bayesian Factors
  Analysers

References
----------
.. [1] `Ghahramani, Zoubin, Matthew J Beal, Gatsby Computational, and Neuroscience
  Unit. “Variational Inference for Bayesian Mixtures of Factor Analysers.” NIPS,
  1999. <http://www.gatsby.ucl.ac.uk/publications/papers/06-2000.pdf>`_
.. [2] `Bishop, Christopher M. “Variational Principal Components,” 1999.
  <http://digital-library.theiet.org/content/conferences/10.1049/cp_19991160.>`_
.. [3] `Beal, Matthew J. “Variational Algorithms For Approximate Bayesian
  Inference,” 2003. <http://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0CC0QFjAB&url=http%3A%2F%2Fwww.cse.buffalo.edu%2Ffaculty%2Fmbeal%2Fpapers%2Fbeal03.pdf&ei=ChT6U4mOIYbiavLXgZAP&usg=AFQjCNE2LgZHagMBM7pJACGSsk4l0jgK9w&sig2=c0f_fiXWy4DekYfh6wimLA&bvm=bv.73612305,d.d2s>`_
