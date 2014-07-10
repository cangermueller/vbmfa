import numpy as np
from scipy.special import digamma
import ipdb

class VbFa(object):
    def __init__(self, hyper, y):
        self.HYPER = hyper
        self.Y = y
        self.P = self.HYPER.P
        self.Q = self.HYPER.Q
        self.N = self.Y.shape[1]
        self.q_nu = Nu(self.Q)
        self.q_mu = Mu(self.P)
        self.q_lambda = Lambda(self.P, self.Q)
        self.q_x = X(self.Q, self.N)

    def update_nu(self):
        self.q_nu.update(self.HYPER, self.q_lambda)

    def update_lambda(self, **kwargs):
        self.q_lambda.update(self.HYPER, self.q_mu, self.q_nu, self.q_x, self.Y, **kwargs)

    def update_x(self):
        self.q_x.update(self.HYPER, self.q_lambda, self.q_mu, self.Y)

    def update_mu(self, **kwargs):
        self.q_mu.update(self.HYPER, self.q_lambda, self.q_x, self.Y, **kwargs)

    def update(self, names=['nu', 'lambda', 'x', 'mu'], **kwargs):
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

    def q(self, name):
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
        self.q_mu.mean = self.Y.mean(1)

    def fit(self, maxit=10, eps=0.0, verbose=True):
        self.init()
        i = 0
        while i < maxit:
            mse = self.mse()
            self.update()
            i += 1
            if verbose:
                print 'Iteration {:d}: {:.3f}'.format(i, self.mse())
            if not mse is None and mse - self.mse() < eps:
                break
        return i

    def x_to_y(self, x=None):
        if x is None:
            x = self.q_x.mean
        return self.q_lambda.mean.dot(x) + self.q_mu.mean[:, np.newaxis]

    def y_to_x(self, y=None):
        if y is None:
            y = self.Y
        y = y - np.mean(y, 1)[:, np.newaxis]
        return self.q_x.cov.dot(np.multiply(self.q_lambda.transpose(), self.HYPER.psi)).dot(y)

    def mse(self):
        return np.linalg.norm(self.Y - self.x_to_y())


class Hyper(object):
    def __init__(self, P, Q=None):
        self.P = P
        self.Q = P if Q is None else Q
        self.a = 1.0
        self.b = 1.0
        self.mu = np.zeros(self.P)
        self.nu = np.ones(self.P)
        self.psi = np.ones(self.P) * 10.0  # precision

    def __str__(self):
        s = '\na: {:f}, b: {:f}'.format(self.a, self.b)
        s += '\nmu: {:s}'.format(self.mu.__str__())
        s += '\nnu: {:s}'.format(self.nu.__str__())
        s += '\npsi: {:s}'.format(self.psi.__str__())
        return s

class Nu(object):

    def __init__(self, Q):
        self.Q = Q
        self.init_rnd()

    def init_rnd(self):
        self.a = 1.0
        self.b = np.ones(self.Q)

    def update(self, hyper, q_lambda):
        self.a = hyper.a + 0.5 * hyper.P
        self.b.fill(hyper.b)
        self.b += 0.5 * (np.sum(q_lambda.mean**2, 0) + np.diag(np.sum(q_lambda.cov, 0)))
        assert np.all(self.b > hyper.b)

    def __str__(self):
        return 'a: {:f}\nb: {:s}'.format(self.a, self.b.__str__())

    def expectation(self):
        return self.a / self.b


class Mu(object):

    def __init__(self, P):
        self.P = P
        self.init_rnd()

    def init_rnd(self):
        self.mean = np.random.normal(loc=0.0, scale=1e-3, size=self.P)
        self.cov = np.ones(self.P)

    def __str__(self):
        return 'mean:\n{:s}\ncov:\n{:s}'.format(self.mean.__str__(), self.cov.__str__())

    def update(self, hyper, q_lambda, q_x, y, x_s=None):
        if x_s is None:
            x_s = np.ones(q_x.N)
        # cov
        self.cov = hyper.nu + hyper.psi * np.sum(x_s)
        self.cov = self.cov**-1
        # mean
        self.mean = np.multiply(hyper.psi, (y - q_lambda.mean.dot(q_x.mean)).dot(x_s)) + np.multiply(hyper.mu, hyper.nu)
        self.mean = np.multiply(self.cov, self.mean)



class Lambda(object):

    def __init__(self, P, Q):
        self.P = P
        self.Q = Q
        self.init_rnd()

    def init_rnd(self):
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
        # Mean
        self.mean = np.empty((self.P, self.Q))
        for p in range(self.P):
            w = np.multiply(x_s, y[p] - q_mu.mean[p])
            self.mean[p] = hyper.psi[p] * self.cov[p].dot(q_x.mean.dot(w))


class X(object):

    def __init__(self, Q, N):
        self.Q = Q
        self.N = N
        self.init_rnd()

    def init_rnd(self):
        self.mean = np.random.normal(loc=0.0, scale=1.0, size=self.Q * self.N).reshape(self.Q, self.N)
        self.cov = np.eye(self.Q)

    def update(self, hyper, q_lambda, q_mu, y):
        # cov
        self.cov = np.eye(self.Q) + np.multiply(q_lambda.mean.transpose(), hyper.psi).dot(q_lambda.mean)
        for p in range(len(hyper.psi)):
            self.cov += hyper.psi[p] * q_lambda.cov[p]
        self.cov = np.linalg.inv(self.cov)
        # mean
        self.mean = self.cov.dot(np.multiply(q_lambda.mean.transpose(), hyper.psi).dot(y - q_mu.mean[:, np.newaxis]))

    def __str__(self):
        return 'mean:\n{:s}\ncov:\n{:s}'.format(self.mean.transpose().__str__(), self.cov.__str__())
