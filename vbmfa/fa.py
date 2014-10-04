"""Variational Bayesian Factor Analyser.

Implementation of a single factor analyser.
Model parameters are inferred by variational Bayes.
"""


import numpy as np
from scipy.special import digamma


class VbFa(object):
    """Variational Bayesian Factor Analyser

    Takes a :math:`p \\times n` data matrix :math:`y` with :math:`n` samples
    :math:`y_i` of dimension :math:`p`, and describes them as a linear
    combination of :math:`q` latent factors:

    .. math::

        P(y_i|\Lambda, x_i, \Psi) = N(y_i|\Lambda x_i + \mu, \Psi)

    :math:`\\Lambda` is the :math:`p \\times q` factor matrix, :math:`x_i` the
    :math:`q` dimensional representation of :math:`y_i`, :math:`\\mu` the mean
    vector, and :math:`\\Psi` the diagonal noise matrix.

    Parameters
    ----------
    y : :py:class:`numpy.ndarray`
        Data matrix with samples in columns and features in rows
    q : int
        Dimension of low-dimensional space (# factors)
    hyper : :py:class:`fa.Hyper`

    Attributes
    ----------
    Y : :py:class:`numpy.ndarray`
        Data matrix with samples in columns and features in rows
    P : int
        Dimension of high-dimensional space
    Q : int
        Dimension of low-dimensional space (# factors)
    N : int
        # Samples
    hyper : :py:class:`fa.Hyper`
        Hyperparameters
    q_nu : :py:class:`fa.Nu`
        Nu distribution
    q_mu : :py:class:`fa.Mu`
        Mu distribution
    q_lambda : :py:class:`fa.Lambda`
        Lambda distribution
    q_x : :py:class:`fa.X`
        X distribution

    Examples
    --------
    .. code:: python

        fa = VbFa(data, q=2)
        fa.fit()
        print(fa.q_lambda.mean)
        print(fa.q_x.mean)
    """

    def __init__(self, y, q=None, hyper=None):
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
        and return number of update iterations.

        Parameters
        ----------
        maxit : int
            Maximum number of update iterations
        eps : float
            Stop if change in MSE is below eps
        verbose : bool
            Print statistics

        Returns
        -------
        num_it : int
            Number of iterations
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
                print('{:d}: {:.3f}'.format(i, mse_new))
            if delta < eps:
                break
        return i

    def mse(self):
        """Compute mean squared error (MSE) between original data and
        reconstructed data.

        Returns
        -------
        mse : float
            Mean squared error

        """
        return np.linalg.norm(self.Y - self.x_to_y())
        self.q_x = X(self.Q, self.N)

    def x_to_y(self, x=None):
        """Reconstruct data from low-dimensional representation.

        Parameters
        ----------
        x : :py:class:`numpy.ndarray`
            low-dimensional representation of the data

        Returns
        -------
        y : :py:class:`numpy.ndarray`
            High-dimensional representation
        """
        if x is None:
            x = self.q_x.mean
        return self.q_lambda.mean.dot(x) + self.q_mu.mean[:, np.newaxis]

    def q(self, name):
        """Return distribution q with the given name.

        Parameters
        ----------
        name : str
            Name of the q distribution
        """
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
        """Update nu distribution."""
        self.q_nu.update(self.HYPER, self.q_lambda)

    def update_lambda(self, x_s=None):
        """Update lambda distribution.

        Parameters
        ----------
        x_s : :py:class:`numpy.ndarray`
            sample weights
        """
        self.q_lambda.update(self.HYPER, self.q_mu, self.q_nu, self.q_x,
                             self.Y, x_s=x_s)

    def update_x(self):
        """Update x distribution."""
        self.q_x.update(self.HYPER, self.q_lambda, self.q_mu, self.Y)

    def update_mu(self, x_s=None):
        """Update mu distribution.

        Parameters
        ----------
        x_s : :py:class:`numpy.ndarray`
            sample weights
        """
        self.q_mu.update(self.HYPER, self.q_lambda, self.q_x, self.Y, x_s=x_s)

    def update(self, names=['lambda', 'x', 'nu', 'mu'], **kwargs):
        """Update all distributions once in the given order.

        Parameters
        ----------
        names : list
            Names of distribution to be updated
        """
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

    def variance_explained(self, sort=False, norm=True):
        """Compute variance explained by factors.

        Parameters
        ----------
        sort : bool
            Sort variance explained in descending order
        norm : bool
            Normalize variance explained to sum up to one

        Returns
        -------
        variance_explained : float
            Variance explained
        """
        ve = np.array([l.dot(l) for l in self.q_lambda.mean.T])
        if sort:
            ve = np.sort(ve)[::-1]
        if norm:
            ve /= ve.sum()
        return ve

    def factors_order(self):
        """Return order of factors by their fraction of variance explained."""
        ve = self.variance_explained()
        return ve.argsort()[::-1]

    def permute(self, order):
        """Permute factors in the given order.

        Parameters
        ----------
        order : :py:class:`numpy.ndarray`
            Permutation order
        """
        self.q_lambda.permute(order)
        self.q_nu.permute(order)
        self.q_x.permute(order)

    def order_factors(self):
        """Orders factors by the fraction of variance explained."""
        self.permute(self.factors_order())


class Hyper(object):
    """Class for model hyperparameters.

    Parameters
    ----------
    p : int
        Dimension of the high-dimensional space
    q : int
        Dimension of the low-dimensional space

    Attributes
    ----------
    P : int
        Dimension of the high-dimensional space
    Q : int
        Dimension of the low-dimensional space
    a : float
        Alpha parameter of gamma prior over factor matrix
    b : float
        Beta parameter of gamma prior over factor matrix
    mu : :py:class:`numpy.ndarray`
        P dimensional mean vector of normal prior over mu vector
    nu : :py:class:`numpy.ndarray`
        P dimensional precision vector of diagonal mu covariance matrix
    psi : :py:class:`numpy.ndarray`
        P dimensional precision vector of diagonal noise covariance matrix
    """

    def __init__(self, p, q=None):
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

    Dirichlet distribution over factor matrix.

    Parameters
    ----------
    q : int
        Rank (# columns) of factor matrix

    Attributes
    ----------
    Q : int
        Rank (# columns) of factor matrix
    a : float
        Alpha parameter of Dirichlet distribution
    b : float
        Beta parameter of Dirichlet distribution
    """

    def __init__(self, q):
        self.Q = q
        self.init()

    def init(self):
        """Initialize parameters."""
        self.a = 1.0
        self.b = np.ones(self.Q)

    def update(self, hyper, q_lambda):
        """Update parameter.

        Parameters
        ----------
        hyper : :py:class:`fa.Hyper`
            Hyperparameters
        q_lambda : :py:class:`fa.Lambda`
            Factor matrix
        """
        self.a = hyper.a + 0.5 * hyper.P
        self.b.fill(hyper.b)
        self.b += 0.5 * (np.sum(q_lambda.mean**2, 0) + np.diag(np.sum(q_lambda.cov, 0)))
        assert np.all(self.b > hyper.b)

    def __str__(self):
        return 'a: {:f}\nb: {:s}'.format(self.a, self.b.__str__())

    def expectation(self):
        """Return expectation of Dirichlet distribution."""
        return self.a / self.b

    def permute(self, order):
        """Permute factors in the given order.

        Parameters
        ----------
        order : :py:class:`numpy.ndarray`
            Permutation order
        """
        self.b = self.b[order]


class Mu(object):
    """Mu factor class.

    Normal distribution over mu with diagonal covariance matrix.

    Parameters
    ----------
    p : int
        dimension of mu vector

    Attributes
    ----------
    P : int
        dimension of mu vector
    mean : :py:class:`np.ndarray`
        mean of Normal distribution
    cov : :py:class:`np.ndarray`
        diagonal of covariance matrix
    """

    def __init__(self, p):
        self.P = p
        self.init()

    def init(self):
        """Initialize parameters."""
        self.mean = np.random.normal(loc=0.0, scale=1e-3, size=self.P)
        self.cov = np.ones(self.P)

    def __str__(self):
        return 'mean:\n{:s}\ncov:\n{:s}'.format(self.mean.__str__(), self.cov.__str__())

    def update(self, hyper, q_lambda, q_x, y, x_s=None):
        """Update parameters.

        Parameters
        ----------
        hyper : :py:class:`fa.Hyper`
            Hyperparameters
        q_lambda : :py:class:`fa.Lambda`
            Factor matrix
        q_x : :py:class:`fa.X`
            Factor loadings matrix
        x_s : :py:class:`numpy.ndarray`
            Sample weights
        """
        if x_s is None:
            x_s = np.ones(q_x.N)
        # cov
        self.cov = hyper.nu + hyper.psi * np.sum(x_s)
        self.cov = self.cov**-1
        # mean
        self.mean = np.multiply(hyper.psi, (y - q_lambda.mean.dot(q_x.mean)).dot(x_s)) + np.multiply(hyper.mu, hyper.nu)
        self.mean = np.multiply(self.cov, self.mean)


class Lambda(object):
    """Lambda factor matrix class.

    Normal distributions over P rows of lambda matrix.

    Parameters
    ----------
    p : int
        # Rows of lambda matrix
    q : int
        # Columns of lambda matrix

    Attributes
    ----------
    P : int
        # Rows of lambda matrix
    Q : int
        # Columns of lambda matrix
    mean : :py:class:`numpy.ndarray`
        Mean of lambda matrix
    cov : :py:class:`numpy.ndarray`
        P QxQ covariance matrices for all rows
    """

    def __init__(self, p, q):
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
        """Update parameters.

        Parameters
        ----------
        hyper : :py:class:`fa.Hyper`
            Hyperparameters
        q_mu : :py:class:`fa.Mu`
            Mu distribution
        q_nu : :py:class:`fa.Nu`
            Nu distribution
        q_x : :py:class:`fa.X`
            X distribution
        y : :py:class:`numpy.ndarray`
            Data matrix
        x_s : :py:class:`numpy.ndarray`
            Sample weights
        """
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

    def permute(self, order):
        """Permute factors in the given order.

        Parameters
        ----------
        order : :py:class:`numpy.ndarray`
            Permutation order
        """
        self.mean = self.mean[:, order]
        for p in range(self.P):
            self.cov[p] = self.cov[p, order, :]
            self.cov[p] = self.cov[p, :, order]


class X(object):
    """X factor class.

    Normal distributions over N columns of X matrix.

    Parameters
    ----------
    q : int
        # Rows of X matrix
    n : int
        # Columns (# samples) of X matrix

    Attributes
    ----------
    Q : int
        # Rows of X matrix
    N : int
        # Columns (# samples) of X matrix
    mean : :py:class:`numpy.ndarray`
        QxN mean of X matrix
    cov : :py:class:`numpy.ndarray`
        QxQ covariance matrix shared for all N columns (samples)
    """

    def __init__(self, q, n):
        self.Q = q
        self.N = n
        self.init()

    def init(self):
        """Initialize parameters."""
        self.mean = np.random.normal(loc=0.0, scale=1.0, size=self.Q * self.N).reshape(self.Q, self.N)
        self.cov = np.eye(self.Q)

    def update(self, hyper, q_lambda, q_mu, y):
        """Update parameters.

        Parameters
        ----------
        hyper : :py:class:`fa.Hyper`
            Hyperparameters
        q_lambda : :py:class:`fa.Lambda`
            Lambda distribution
        q_mu : :py:class:`fa.Mu`
            Mu distribution
        y : :py:class:`numpy.ndarray`
            Data matrix
        """
        # cov
        self.cov = np.eye(self.Q) + np.multiply(q_lambda.mean.transpose(), hyper.psi).dot(q_lambda.mean)
        for p in range(len(hyper.psi)):
            self.cov += hyper.psi[p] * q_lambda.cov[p]
        self.cov = np.linalg.inv(self.cov)
        # mean
        self.mean = self.cov.dot(np.multiply(q_lambda.mean.transpose(), hyper.psi).dot(y - q_mu.mean[:, np.newaxis]))

    def __str__(self):
        return 'mean:\n{:s}\ncov:\n{:s}'.format(self.mean.transpose().__str__(), self.cov.__str__())

    def permute(self, order):
        """ Permute factors in the given order.

        Parameters
        ----------
        order : :py:class:`numpy.ndarray`
            Permutation order
        """
        self.mean = self.mean[order, :]
        self.cov = self.cov[order, :]
        self.cov = self.cov[:, order]
