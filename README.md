vbmfa
=====

Python implementation of a Variational Bayesian Mixture of Factor Analysers.

## Introduction

Factor analysis (FA) is a method for dimensionality reduction, similar to principle component analysis (PCA), singular value decomposition (SVD), or independent component analysis (ICA). Applications include visualization, image compression, or feature learning. A mixture of factor analysers consists of several factor analysers, and allows both dimensionality reduction and clustering. Variational Bayesian learning of model parameters prevents overfitting compared with maximum likelihood methods such as expectation maximization (EM), and allows to learn the dimensionality of the lower dimensional subspace by automatic relevance determination (ARD).

### Reference:
* Ghahramani, Zoubin, Matthew J Beal, Gatsby Computational, and Neuroscience Unit. “Variational Inference for Bayesian Mixtures of Factor Analysers.” NIPS, 1999.
* Bishop, Christopher M. “Variational Principal Components,” 1999. http://digital-library.theiet.org/content/conferences/10.1049/cp_19991160.
* Beal, Matthew J. “Variational Algorithms For Approximate Bayesian Inference,” 2003.

## Content

The module `vbmfa` contains two major classes:
* `VbFa`, which is a simple factor analyser
* `VbMfa`, which is a mixture of factor analysers.

Usage examples can be found in the folder `notebooks` for ipython notebooks:
* [VbFa](http://nbviewer.ipython.org/github/cangermueller/vbmfa/blob/master/notebooks/140709_vbfa.ipynb): `VbFa` examples
* [VbMfa](http://nbviewer.ipython.org/github/cangermueller/vbmfa/blob/master/notebooks/140709_vbmfa.ipynb): `VbMfa` examples
