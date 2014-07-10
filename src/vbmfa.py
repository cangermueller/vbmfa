import numpy as np
from scipy.special import digamma
import sklearn.cluster
import vbfa
import pdb


class VbMfa(object):
    def __init__(self, hyper, y):
        self.HYPER = hyper
        self.Y = y
        self.P = self.HYPER.P
        self.Q = self.HYPER.Q
        self.S = self.HYPER.S
        self.N = self.Y.shape[1]
        self.fas = [vbfa.VbFa(self.HYPER, y) for s in range(self.S)]
        self.q_pi = Pi(self.HYPER.S)
        self.q_s = S((self.HYPER.S, self.N))

    def update_pi(self):
        self.q_pi.update(self.HYPER, self.q_s)

    def update_s(self):
        self.q_s.update(self.HYPER, self.q_pi, self.fas, self.Y)

    def update_fas(self):
        for s in range(self.S):
            self.fas[s].update(x_s=self.q_s[s])

    def update(self):
        self.update_fas()
        self.update_s()
        self.update_pi()

    def x_to_y(self):
        y = np.zeros((self.P, self.N))
        for s in range(self.S):
            y += np.multiply(self.fas[s].x_to_y(), self.q_s[s])
        return y

    def mse(self):
        return np.linalg.norm(self.Y - self.x_to_y())

    def init_kmeans(self):
        if self.S == 1:
            self.fas[0].init()
        else:
            km = sklearn.cluster.KMeans(self.S)
            km.fit(self.Y.transpose())
            for s in range(self.S):
                self.fas[s].q_mu.mean[:] = km.cluster_centers_[s]
            self.update_s()

    def fit(self, maxit=10, eps=0.0, verbose=True):
        self.init_kmeans()
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

    def __str__(self):
        means = np.empty((self.P, self.S))
        for s in range(self.S):
            means[:, s] = self.fas[s].q_mu.mean
        return 'means:\n{:s}'.format(means)

    def map_x(self):
        x = np.empty((self.Q, self.N))
        for s in range(self.S):
            x += np.multiply(self.fas[s].q_x.mean, self.q_s[0])
        return x



class Hyper(vbfa.Hyper):
    def __init__(self, P, Q, S=1):
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
    def __init__(self, S):
        self.S = S
        self.init_rnd()

    def init_rnd(self):
        self.alpha = np.random.normal(loc=1.0, scale=1e-3, size=self.S)

    def update(self, hyper, q_s):
        self.alpha = hyper.alpha * hyper.m + np.sum(q_s, 1)

    def __str__(self):
        return 'alpha: {:s}'.format(self.alpha.__str__())

    def expectation(self):
        return self.alpha / float(sum(self.alpha))



class S(np.ndarray):
    def __init__(self, (S, N)):
        self.S = S
        self.N = N
        self.init_rnd()

    def init_rnd(self):
        # self[:] = np.random.normal(1.0, 10, self.S * self.N).reshape(self.S, self.N)
        self[:] = np.random.rand(self.S, self.N)
        self[:] = self / np.sum(self, 0)

    def update_s(self, hyper, q_lambda, q_mu, q_pi_alpha, q_x, y):
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

    def update(self, hyper, q_pi, fas, y):
        for s in range(len(fas)):
            self[s, :] = self.update_s(hyper, fas[s].q_lambda, fas[s].q_mu, q_pi.alpha[s], fas[s].q_x, y)
        self[:] = self - self.max(0)
        self[:] = np.maximum(np.exp(self), 1e-5)
        self[:] = self / np.sum(self, 0)

    def __str__(self):
        t = np.array(self.transpose())
        return str(t)
