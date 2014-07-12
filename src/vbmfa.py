"""Variational Bayesian Mixture of Factor Analyser.

Implementation of a mixture of factor analysers.
Model parameters are inferred by variational Bayes.
"""

import numpy as np
from scipy.special import digamma
import sklearn.cluster
import vbfa
import numpy.testing as npt
import pdb


class VbMfa(object):
    """Variational Bayesian Mixture of Factor Analysers.

    Factorizes Y = L_s X_s + mu_s, where Y is the input data matrix,
    L_s a factor loading matrix, X_s a factor matrix, mu_s a mean vector, and
    s=1,...,S, where S is the total number of factor analysers (=components).
    Given input matrix Y and rank Q of L_s and X_s, uses variational Bayes to
    infer model parameters.
    Example:
    > mfa = VbMfa(data, q=2, s=2)
    > mfa.fit()
    > print mfa.fas[0].q_lambda.mean // factor loading matrix L of component 0
    > print mfa.fas[1].q_x.mean // factor matrix X of component 1
    > print mfa.q_s // component indicators
    > print mfa.q_pi // component distribution
    """
    def __init__(self, y, q=None, s=1, hyper=None):
        """Construct VbMfa instance.

        Y -- data matrix with samples in columns and features in rows
        P -- dimension of the high-dimensional space
        Q -- dimension of the low-dimensional space
        S -- # factor analysers (# components)
        N -- # samples
        HYPER -- model hyperparameters
        fas -- list of S VbFa factor analyser instances
        q_pi -- Pi factor
        q_s -- S factor
        """
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
        self.fas = [vbfa.VbFa(self.Y, self.Q, hyper=self.HYPER) for s in range(self.S)]
        self.q_pi = Pi(self.S)
        self.q_s = S((self.S, self.N))

    def fit(self, maxit=10, eps=0.0, **kwargs):
        """Fit model parameters by updating factors for several iterations.

        maxit -- maximum number of update iterations
        eps -- stop if change in mse is below eps
        and return number of update iterations until convergence.
        **kwargs -- keyword arguments for converge()
        """
        self.init()
        return self.converge(self.update, maxit=maxit, eps=eps, **kwargs)

    def fit_highdim(self, maxit=10, eps=0.0, verbose=False):
        """Fit model parameters to high-dimensional data (P large).

        Same as fit(), but with special update heuristic which might work
        better for high-dimensional data.
        """
        if verbose:
            print 'Initialization ...'
        self.init()
        if verbose:
            print 'Initialization components ...'
        self.converge(self.update_fas, maxit=maxit, eps=0.5, verbose=verbose)
        if verbose:
            print 'Initialization assignments ...'
        self.converge(lambda: self.update_s_pi(damp=0.8), maxit=maxit, eps=0.0, verbose=verbose)
        if verbose:
            print 'Fitting ...'
        self.converge(lambda: self.update(damp=0.8), maxit=maxit, eps=eps, verbose=verbose)

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

        Uses k-means to estimate cluster centers, which are used to initialize q_mu.
        q_s is initialized by the distance of samples Y[:n] to the cluster
        centers.
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
        """Update all factors once."""
        self.update_fas()
        self.update_s(**kwargs)
        self.update_pi()

    def converge(self, fun, maxit=1, eps=0.5, verbose=False):
        """Call function fun until MSE converges and return # iterations.

        maxit -- maximum # iterations
        eps -- stop if change in MSE is below eps
        verbose -- print statistics
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
                print '{:d}: {:.3f}'.format(it, mse_new)
        return it

    def update_pi(self):
        """Update pi factor."""
        self.q_pi.update(self.HYPER, self.q_s)

    def update_s(self, **kwargs):
        """Update s factor."""
        self.q_s.update(self.HYPER, self.q_pi, self.fas, self.Y, **kwargs)

    def update_s_pi(self, **kwargs):
        """Update s and pi factor."""
        self.update_s(**kwargs)
        self.update_pi()

    def update_fas(self):
        """Update all factor analysers."""
        for s in range(self.S):
            self.fas[s].update(x_s=self.q_s[s])

    def __str__(self):
        means = np.empty((self.P, self.S))
        for s in range(self.S):
            means[:, s] = self.fas[s].q_mu.mean
        return 'means:\n{:s}'.format(means)


class Hyper(vbfa.Hyper):
    """Class for model hyperparameters."""
    def __init__(self, P, Q, S=1):
        """Construct Hyper instance.

        Inherits from vbfa.VbFa.
        S -- number of factor analysers
        alpha -- strength of Dirichlet prior on pi
        m -- distribution of Dirichlet prior on pi
        """
        super(Hyper, self).__init__(P, Q)
        self.S = S
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
    """
    def __init__(self, S):
        """Construct Pi instance.

        S -- # components
        alpha -- S dimensional vector of Dirichlet parameters.
        """
        self.S = S
        self.init()

    def init(self):
        """Initialize parameters."""
        self.alpha = np.random.normal(loc=1.0, scale=1e-3, size=self.S)

    def update(self, hyper, q_s):
        """Update parameters."""
        self.alpha = hyper.alpha * hyper.m + np.sum(q_s, 1)

    def __str__(self):
        return 'alpha: {:s}'.format(self.alpha.__str__())

    def expectation(self):
        """Return expectation of Dirichlet distribution."""
        return self.alpha / float(sum(self.alpha))


class S(np.ndarray):
    """S factor class.

    SxN matrix, where s_i,j is the probability that sample j
    belongs to component i.
    """

    def __init__(self, (S, N)):
        """Construct S instance.

        Inherits from numpy.ndarray
        S -- # components
        N -- # samples
        """
        self.S = S
        self.N = N
        self.init()

    def init(self):
        """Initialize parameters."""
        self[:] = np.random.normal(10.0, 1e-1, self.S * self.N).reshape(self.S, self.N)
        self[:] = np.maximum(0.0, self[:])
        self[:] = self / np.sum(self, 0)

    def update_s(self, hyper, q_lambda, q_mu, q_pi_alpha, q_x, y):
        """Update parameters of component s."""
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
        """Update parameters of all components."""
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
