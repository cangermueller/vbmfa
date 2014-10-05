"""Variational Bayesian Mixture of Factor Analysers.

Implementation of a mixture of factor analysers.
Model parameters are inferred by variational Bayes.
"""

import numpy as np
from scipy.special import digamma
import sklearn.cluster
import numpy.testing as npt
import vbmfa.fa as vbmfa_fa
# Use the following line instead for sphinx
# import fa as vbmfa_fa


class VbMfa(object):
    """Variational Bayesian Mixture of Factor Analysers.

    Takes a :math:`p \\times n` data matrix :math:`y` with :math:`n` samples
    :math:`y_i` of dimension :math:`p`, and describes them as a mixture of
    :math:`s` components, i.e. factor analysers, each of which has :math:`q`
    latent factors:

    .. math::

        P(y_i|\\theta) = \sum_s P(s) P(y_i|\\theta_s)

    Here, :math:`\\theta_s` are the parameters of component :math:`s`
    (see :mod:`fa`). Efficient variational Bayes is used to infer the
    distribution over all model parameters.

    Parameters
    ----------
    y : :py:class:`numpy.ndarray`
        Data matrix with samples in columns and features in rows
    q : int
        Dimension of the low-dimensional space (# factors)
    s : int
        # components
    hyper : :py:class:`mfa.Hyper`
        Model hyperparameters

    Attributes
    ----------
    Y : :py:class:`numpy.ndarray`
        Data matrix with samples in columns and features in rows
    P : int
        Dimension of the high-dimensional space
    Q : int
        Dimension of the low-dimensional space (# factors)
    S : int
        # components
    N : int
        # Samples
    HYPER : :py:class:`fa.Hyper`
        Model hyperparameters
    fas : list
        List of S VbFa factor analyser instances
    q_pi : :py:class:`mfa.Pi`
        Distribution over Pi
    q_s : :py:class:`mfa.S`
        Distribution over component indicators s

    Examples
    --------
    .. code:: python

        mfa = VbMfa(data, q=2, s=2)
        mfa.fit()
        print(mfa.fas[0].q_lambda.mean)
        print(mfa.fas[1].q_x.mean)
        print(mfa.q_s)
        print(mfa.q_pi)

    Note
    ----
    The term `component` means the same as `factor analyser`.

    """
    def __init__(self, y, q=None, s=1, hyper=None):
        self.Y = y
        self.P = self.Y.shape[0]
        self.Q = self.P if q is None else q
        self.S = s
        self.N = self.Y.shape[1]
        if hyper is None:
            self.HYPER = Hyper(self.P, self.Q, self.S)
        else:
            self.HYPER = hyper
        self.Y = y
        self.fas = [vbmfa_fa.VbFa(self.Y, self.Q, hyper=self.HYPER) for s in range(self.S)]
        self.q_pi = Pi(self.S)
        self.q_s = S((self.S, self.N))

    def fit(self, maxit=10, eps=0.0, **kwargs):
        """Fit model parameters by updating paramters until convergence.

        Parameters
        ----------
        maxit : int
            Maximum number of update iterations
        eps : float
            Stop if change in mse is below eps
        kwargs : dict
            Keyword arguments passed to :py:func:`converge`
        verbose : bool
            Print statistics after each iteration

        Returns
        -------
        num_it : int
            Number of iterations
        """
        self.init()
        num_it = self.converge(self.update, maxit=maxit, eps=eps, **kwargs)
        return num_it

    def fit_highdim(self, maxit=10, eps=0.0, verbose=False, **kwargs):
        """Fit model parameters to high-dimensional data (P large).

        Same as fit(), but with special update heuristic which might work
        better for high-dimensional data.
        """
        if verbose:
            print('Initialization ...')
        self.init()
        if verbose:
            print('Initialization components ...')
        self.converge(self.update_fas, maxit=maxit, eps=0.5, **kwargs)
        if verbose:
            print('Initialization assignments ...')
        self.converge(lambda: self.update_s_pi(damp=0.8), maxit=maxit, eps=0.0,
                      **kwargs)
        if verbose:
            print('Fitting ...')
        num_it = self.converge(lambda: self.update(damp=0.8), maxit=maxit,
                               eps=eps, **kwargs)
        self.order_factors()
        return num_it

    def mse(self):
        """Compute mean squared error (MSE) between original data and
        reconstructed data."""
        return np.linalg.norm(self.Y - self.x_to_y())

    def x_to_y(self):
        """Reconstruct data from low-dimensional representation."""
        y = np.zeros((self.P, self.N))
        for s in range(self.S):
            y += np.multiply(self.fas[s].x_to_y(), self.q_s[s])
        return y

    def init(self):
        """Initialize factors for fitting.

        Uses k-means to estimate cluster centers, which are used to initialize
        q_mu.  q_s is initialized by the distance of samples Y[:n] to the
        cluster centers.
        """
        if self.S == 1:
            self.fas[0].init()
        else:
            km = sklearn.cluster.KMeans(self.S)
            km.fit(self.Y.transpose())
            for s in range(self.S):
                self.fas[s].q_mu.mean[:] = km.cluster_centers_[s]

            # init S
            dist = np.empty((self.S, self.N))
            for s in range(self.S):
                dist[s] = np.linalg.norm(self.Y - km.cluster_centers_[s][:, np.newaxis], axis=0)
            dist = dist.max(0) * 1.5 - dist
            dist /= dist.sum(0)
            self.q_s[:] = dist

    def update(self, **kwargs):
        """Update all factors once.

        Parameters
        ----------
        kwargs: dict
            Passed to :py:func:`VbMfa.update_s`
        """
        self.update_fas()
        self.update_s(**kwargs)
        self.update_pi()

    def converge(self, fun, maxit=1, eps=0.5, verbose=False):
        """Call function fun until MSE converges.

        Parameters
        ----------
        maxit : int
            Maximum # iterations
        eps : float
            Stop if change in MSE is below eps
        verbose : bool
            Print statistics

        Returns
        -------
        num_it : int
            # iterations until convergence
        """
        it = 0
        delta = eps
        while it < maxit and delta >= eps:
            mse_old = self.mse()
            fun()
            mse_new = self.mse()
            delta = mse_old - mse_new
            it += 1
            if verbose:
                print('{:d}: {:.3f}'.format(it, mse_new))
        return it

    def update_pi(self):
        """Update pi factor."""
        self.q_pi.update(self.HYPER, self.q_s)

    def update_s(self, **kwargs):
        """Update s factor.

        Parameters
        ----------
        kwargs : dict
            Passed to :py:func:`S.update`
        """
        self.q_s.update(self.HYPER, self.q_pi, self.fas, self.Y, **kwargs)

    def update_s_pi(self, **kwargs):
        """Update s and pi factor.

        Parameters
        ----------
        kwargs : dict
            Passed to :py:func:`S.update`
        """
        self.update_s(**kwargs)
        self.update_pi()

    def update_fas(self):
        """Update all components."""
        for s in range(self.S):
            self.fas[s].update(x_s=self.q_s[s])

    def __str__(self):
        means = np.empty((self.P, self.S))
        for s in range(self.S):
            means[:, s] = self.fas[s].q_mu.mean
        return 'means:\n{:s}'.format(means)

    def order_factors(self):
        """Order factors of all components."""
        for f in self.fas:
            f.order_factors()


class Hyper(vbmfa_fa.Hyper):
    """Class for model hyperparameters.

    Inherits from :py:class:`fa.Hyper`.

    Parameters
    ----------
    p : int
        Dimension of the high-dimensional space
    q : int
        Dimension of the low-dimensional space
    s : int
        # components

    Attributes
    ----------
    S : int
        # components
    alpha : float
        Strength of Dirichlet prior on pi
    m : :py:class:`numpy.ndarray`
        Distribution of Dirichlet prior on pi
    """
    def __init__(self, p, q, s=1):
        super(Hyper, self).__init__(p, q)
        self.S = s
        self.alpha = 1.0
        self.m = np.empty(self.S)
        self.m.fill(self.S**-1)

    def __str__(self):
        s = super(Hyper, self).__str__()
        s += '\nalpha: {:s}'.format(self.alpha.__str__())
        return s


class Pi:
    """Pi factor class.

    Dirichlet distribution over component probabilities pi.

    Parameters
    ----------
    s : int
        # components

    Attributes
    ----------
    S : int
        # components
    alpha : :py:class:`numpy.ndarray`
        S dimensional vector of Dirichlet parameters.
    """
    def __init__(self, s):
        self.S = s
        self.init()

    def init(self):
        """Initialize parameters."""
        self.alpha = np.random.normal(loc=1.0, scale=1e-3, size=self.S)

    def update(self, hyper, q_s):
        """Update parameters.

        Parameters
        ----------
        hyper : :py:class:`mfa.Hyper`
            Hyperparameters
        q_s : :py:class:`mfa.S`
            Distribution over component indicators s
        """
        self.alpha = hyper.alpha * hyper.m + np.sum(q_s, 1)

    def __str__(self):
        return 'alpha: {:s}'.format(self.alpha.__str__())

    def expectation(self):
        """Compute expectation of Dirichlet distribution.

        Returns
        -------
        exp : float
            Expectation of Dirichlet distribution
        """
        return self.alpha / float(sum(self.alpha))


class S(np.ndarray):
    """S factor class.

    :math:`s \\times n` dimensional matrix :math:`S`, where :math:`s_{ij}` is the
    probability that sample :math:`j` belongs to component :math:`i`.

    Inherits from :py:class:`numpy.ndarray`.

    Parameters
    ----------
    dim : array like
        Shape (s, n) of matrix

    Attributes
    ----------
    S : int
        # components
    N : int
        # samples
    """

    def __init__(self, dim):
        (S, N) = dim
        self.S = S
        self.N = N
        self.init()

    def init(self):
        """Initialize parameters."""
        self[:] = np.random.normal(10.0, 1e-1, self.S * self.N).reshape(self.S, self.N)
        self[:] = np.maximum(0.0, self[:])
        self[:] = self / np.sum(self, 0)

    def update_s(self, hyper, q_lambda, q_mu, q_pi_alpha, q_x, y):
        """Update parameters of component s in log-space.

        Parameters
        ----------
        hyper : :py:class:`mfa.Hyper`
            Hyperparameters
        q_lambda : :py:class:`fa.Lambda`
            Lambda distribution
        q_mu : :py:class:`fa.Mu`
            Mu distribution
        q_pi : :py:class:`mfa.Pi`
            Pi distribution
        q_pi_alpha : :py:class:`numpy.ndarray`
            Alpha parameters of Pi distribution
        q_x : :py:class:`fa.X`
            X distribution
        y : :py:class:`numpy.ndarray`
            Data matrix

        Returns
        -------
        s : :py:class:`numpy.ndarray`
            Updated parameters of N samples of component s in log-space
        """
        # const
        const = digamma(q_pi_alpha) + 0.5 * np.linalg.slogdet(q_x.cov)[1]
        # fit
        fit = y - q_lambda.mean.dot(q_x.mean) - q_mu.mean[:, np.newaxis]
        # fit = -0.5 * np.diag(fit.transpose().dot(np.diag(self.HYPER.psi)).dot(fit))
        fit = -0.5 * np.diag(np.multiply(fit, hyper.psi[:, np.newaxis]).transpose().dot(fit))
        # cov
        lambda_cov = np.empty((hyper.P, hyper.Q))
        for p in range(hyper.P):
            lambda_cov[p] = np.diagonal(q_lambda.cov[p])
        cov = (q_lambda.mean**2 + lambda_cov).dot(np.diag(q_x.cov)[:, np.newaxis]) \
            + q_mu.cov[:, np.newaxis] + lambda_cov.dot(q_x.mean**2)
        cov = -0.5 * hyper.psi.dot(cov)
        return const + fit + cov

    def update(self, hyper, q_pi, fas, y, damp=0.0):
        """Update parameters of all components.

        Parameters
        ----------
        hyper : :py:class:`mfa.Hyper`
            Hyperparameters
        q_pi : :py:class:`mfa.Pi`
            Pi distribution
        fas : list
            List of all components
        y : :py:class:`numpy.ndarray`
            Data matrix
        damp : float
            Damping factor: reuse ``damp`` % of old parameters
        """
        new = np.empty((self.S, self.N))
        for s in range(len(fas)):
            new[s, :] = self.update_s(hyper, fas[s].q_lambda, fas[s].q_mu, q_pi.alpha[s], fas[s].q_x, y)
        new = new - new.max(0)
        new = np.maximum(np.exp(new), 1e-1)
        new = new / np.sum(new, 0)
        self[:] = damp * self[:] + (1 - damp) * new
        self[:] = self[:] / np.sum(self, 0)
        npt.assert_almost_equal(self.sum(0), 1.0)

    def __str__(self):
        t = np.array(self.transpose())
        return str(t)
