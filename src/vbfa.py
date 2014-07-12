"""Variational Bayesian Factor Analyser.

Implementation of a single factor analyser.
Model parameters are inferred by variational Bayes.
"""

import numpy as np
from scipy.special import digamma
import ipdb


class VbFa(object):
    """Variational Bayesian Factor Analyser.

    Factorizes Y = L X + mu, where Y is the input data matrix, L the factor loading
    matrix, X the factor matrix, and mu the mean vector. L and X are rank Q matrices.
    Given input matrix Y and Q, uses variational Bayes to infer L and X.
    Example:
    > fa = VbFa(data, q=2)
    > fa.fit()
    > print fa.q_lambda.mean // factor loading matrix L
    > print fa.q_x.mean // factor matrix X
    """

    def __init__(self, y, q=None, hyper=None):
        """Construct VbFa instance.

        Y -- data matrix with samples in columns and features in rows
        P -- dimension of the high-dimensional space
        Q -- dimension of the low-dimensional space
        N -- # samples
        HYPER -- model hyperparameters
        q_nu -- Nu factor
        q_lambda -- Lambda factor
        q_x -- X factor
        q_mu -- Mu factor
        """
        self.Y = y
        self.P = self.Y.shape[0]
        self.Q = self.P if q is None else q
        self.N = self.Y.shape[1]
        if hyper is None:
            self.HYPER = Hyper(self.P, self.Q)
        else:
            self.HYPER = hyper
        self.q_nu = Nu(self.Q)
        self.q_mu = Mu(self.P)
        self.q_lambda = Lambda(self.P, self.Q)
        self.q_x = X(self.Q, self.N)

    def fit(self, maxit=10, eps=0.0, verbose=False):
        """Fit model parameters by updating factors for several iterations
        and return number of update iterations until convergence.

        maxit -- maximum number of update iterations
        eps -- stop if change in MSE is below eps
        verbose -- print statistics
        """
        self.init()
        i = 0
        while i < maxit:
            mse_old = self.mse()
            self.update()
            mse_new = self.mse()
            delta = mse_old - mse_new
            i += 1
            if verbose:
                print '{:d}: {:.3f}'.format(i, mse_new)
            if delta < eps:
                break
        return i

    def mse(self):
        """Compute mean squared error (MSE) between original data and
        reconstructed data."""
        return np.linalg.norm(self.Y - self.x_to_y())
        self.q_x = X(self.Q, self.N)

    def x_to_y(self, x=None):
        """Reconstruct data from low-dimensional representation."""
        if x is None:
            x = self.q_x.mean
        return self.q_lambda.mean.dot(x) + self.q_mu.mean[:, np.newaxis]

    def q(self, name):
        """Return factor with the given name."""
        if name == 'nu':
            return self.q_nu
        elif name == 'lambda':
            return self.q_lambda
        elif name == 'x':
            return self.q_x
        elif name == 'mu':
            return self.q_mu
        else:
            raise 'q_{:s} unknown!'.format(name)

    def init(self):
        """Initialize factors for fitting."""
        self.q_mu.mean = self.Y.mean(1)

    def update_nu(self):
        """Update nu factor."""
        self.q_nu.update(self.HYPER, self.q_lambda)

    def update_lambda(self, x_s=None):
        """Update lambda factor."""
        self.q_lambda.update(self.HYPER, self.q_mu, self.q_nu, self.q_x, self.Y, x_s=x_s)

    def update_x(self):
        """Update x factor."""
        self.q_x.update(self.HYPER, self.q_lambda, self.q_mu, self.Y)

    def update_mu(self, x_s=None):
        """Update mu factor."""
        self.q_mu.update(self.HYPER, self.q_lambda, self.q_x, self.Y, x_s=x_s)

    def update(self, names=['lambda', 'x', 'nu', 'mu'], **kwargs):
        """Update all factors once in the given order."""
        if type(names) is str:
            if names == 'nu':
                self.update_nu()
            elif names == 'lambda':
                self.update_lambda(**kwargs)
            elif names == 'mu':
                self.update_mu(**kwargs)
            elif names == 'x':
                self.update_x()
        else:
            for name in names:
                self.update(name, **kwargs)


class Hyper(object):
    """Class for model hyperparameters."""

    def __init__(self, p, q=None):
        """Construct Hyper instance.

        P -- dimension of the high-dimensional space
        Q -- dimension of the low-dimensional space
        a -- alpha parameter of gamma prior over factor loadings
        b -- beta parameter of gamma prior over factor loadings
        mu -- P dimensional mean vector of normal prior over mu vector
        nu -- P dimensional precision vector of mu covariance matrix diagonal
        psi -- P dimensional precision vector of noise covariance matrix diagonal
        """
        self.P = p
        self.Q = p if q is None else q
        self.a = 1.0
        self.b = 1.0
        self.mu = np.zeros(self.P)
        self.nu = np.ones(self.P)
        self.psi = np.ones(self.P) * 10.0

    def __str__(self):
        s = '\na: {:f}, b: {:f}'.format(self.a, self.b)
        s += '\nmu: {:s}'.format(self.mu.__str__())
        s += '\nnu: {:s}'.format(self.nu.__str__())
        s += '\npsi: {:s}'.format(self.psi.__str__())
        return s


class Nu(object):
    """Nu factor class.

    Dirichlet distribution over q factor loading components.
    """
    def __init__(self, q):
        """Construct Nu instance.

        Q -- rank (# columns) of factor loading matrix
        a -- alpha parameter of Dirichlet distribution
        b -- beta parameter of Dirichlet distribution
        """
        self.Q = q
        self.init()

    def init(self):
        """Initialize parameters."""
        self.a = 1.0
        self.b = np.ones(self.Q)

    def update(self, hyper, q_lambda):
        """Update parameter."""
        self.a = hyper.a + 0.5 * hyper.P
        self.b.fill(hyper.b)
        self.b += 0.5 * (np.sum(q_lambda.mean**2, 0) + np.diag(np.sum(q_lambda.cov, 0)))
        assert np.all(self.b > hyper.b)

    def __str__(self):
        return 'a: {:f}\nb: {:s}'.format(self.a, self.b.__str__())

    def expectation(self):
        """Return expectation of Dirichlet distribution."""
        return self.a / self.b


class Mu(object):
    """Mu factor class.

    Normal distribution over mu with diagonal covariance matrix.
    """

    def __init__(self, p):
        """Construct Mu instance.

        P -- dimension of mu vector
        mean -- mean of Normal distribution
        cov -- diagonal of covariance matrix
        """
        self.P = p
        self.init()

    def init(self):
        """Initialize parameters."""
        self.mean = np.random.normal(loc=0.0, scale=1e-3, size=self.P)
        self.cov = np.ones(self.P)

    def __str__(self):
        return 'mean:\n{:s}\ncov:\n{:s}'.format(self.mean.__str__(), self.cov.__str__())

    def update(self, hyper, q_lambda, q_x, y, x_s=None):
        """Update parameters."""
        if x_s is None:
            x_s = np.ones(q_x.N)
        # cov
        self.cov = hyper.nu + hyper.psi * np.sum(x_s)
        self.cov = self.cov**-1
        # mean
        self.mean = np.multiply(hyper.psi, (y - q_lambda.mean.dot(q_x.mean)).dot(x_s)) + np.multiply(hyper.mu, hyper.nu)
        self.mean = np.multiply(self.cov, self.mean)


class Lambda(object):
    """Lambda factor class.

    Normal distributions over P rows of lambda matrix.
    """

    def __init__(self, p, q):
        """Construct Lambda instance.

        P -- #rows of lambda matrix
        Q -- #columns of lambda matrix
        mean -- PxQ mean matrix of lambda matrix
        cov -- P QxQ covariance matrices for all rows
        """
        self.P = p
        self.Q = q
        self.init()

    def init(self):
        """Initialize parameters."""
        self.mean = np.random.normal(loc=0.0, scale=1.0, size=self.P * self.Q).reshape(self.P, self.Q)
        self.cov = np.empty((self.P, self.Q, self.Q))
        for p in range(self.P):
            self.cov[p] = np.eye(self.Q)

    def __str__(self, cov=False):
        s = 'mean:\n{:s}'.format(self.mean.__str__())
        if cov:
            for p in range(self.P):
                s += '\ncov[{:d}]:\n{:s}'.format(p, self.cov[p].__str__())
        return s

    def update(self, hyper, q_mu, q_nu, q_x, y, x_s=None):
        """Update parameters."""
        if x_s is None:
            x_s = np.ones(q_x.N)
        # cov
        assert np.all(q_nu.b > 0.0)
        t = np.zeros((self.Q, self.Q))
        for n in range(len(x_s)):
            t += x_s[n] * (np.outer(q_x.mean[:, n], q_x.mean[:, n]) + q_x.cov)
        tt = np.diag(q_nu.expectation())
        self.cov = np.empty((self.P, self.Q, self.Q))
        for p in range(self.P):
            self.cov[p] = tt + hyper.psi[p] * t
            self.cov[p] = np.linalg.inv(self.cov[p])
        # mean
        self.mean = np.empty((self.P, self.Q))
        for p in range(self.P):
            w = np.multiply(x_s, y[p] - q_mu.mean[p])
            self.mean[p] = hyper.psi[p] * self.cov[p].dot(q_x.mean.dot(w))


class X(object):
    """X factor class.

    Normal distributions over N columns of X matrix.
    """

    def __init__(self, q, n):
        """Construct X instance.

        Q -- #rows of X matrix
        N -- #columns (# samples) of X matrix
        mean -- QxN mean matrix of X matrix
        cov -- QxQ covariance matrix shared for all N columns (samples)
        """
        self.Q = q
        self.N = n
        self.init()

    def init(self):
        """Initialize parameters."""
        self.mean = np.random.normal(loc=0.0, scale=1.0, size=self.Q * self.N).reshape(self.Q, self.N)
        self.cov = np.eye(self.Q)

    def update(self, hyper, q_lambda, q_mu, y):
        """Update parameters."""
        # cov
        self.cov = np.eye(self.Q) + np.multiply(q_lambda.mean.transpose(), hyper.psi).dot(q_lambda.mean)
        for p in range(len(hyper.psi)):
            self.cov += hyper.psi[p] * q_lambda.cov[p]
        self.cov = np.linalg.inv(self.cov)
        # mean
        self.mean = self.cov.dot(np.multiply(q_lambda.mean.transpose(), hyper.psi).dot(y - q_mu.mean[:, np.newaxis]))

    def __str__(self):
        return 'mean:\n{:s}\ncov:\n{:s}'.format(self.mean.transpose().__str__(), self.cov.__str__())
