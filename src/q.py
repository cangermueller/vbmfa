import numpy as np
from scipy.special import digamma
import pdb


def is_pd(M):
    return np.all(np.linalg.eigvals(M) > 0)


class Pi:

    def __init__(self, S):
        self.S = S
        self.init_rnd()

    def init_rnd(self):
        self.alpha = np.random.normal(loc=1.0, scale=1e-3, size=self.S)

    def update(self, h, q_s):
        self.alpha = h.alpha * h.m + np.sum(q_s.s, 1)
        assert np.all(self.alpha > h.alpha*h.m)

    def __str__(self):
        return 'alpha: {:s}'.format(self.alpha.__str__())


class Nu:

    def __init__(self, P, Q):
        self.P = P
        self.Q = Q
        self.init_rnd()

    def init_rnd(self):
        self.a = 1.0
        self.b = np.ones(self.Q)

    def update(self, h, q_lm):
        self.a = h.a+0.5*self.P
        self.b.fill(h.b)
        self.b += 0.5*(np.sum(q_lm.l.mean**2, 0)+np.diag(np.sum(q_lm.cov, 0))[:-1])
        assert np.all(self.b > h.b)

    def __str__(self):
        return 'a: {:f}\nb: {:s}'.format(self.a, self.b.__str__())

    def expectation(self):
        return self.a/self.b



class Mu:

    def __init__(self, P):
        self.P = P
        self.init_rnd()

    def init_rnd(self):
        self.mean = np.random.normal(loc=0.0, scale=0.1, size=self.P)
        self.pre = np.ones(self.P)
        self.cov = 1/self.pre

    def __str__(self):
        return 'mean:\n{:s}\ncov:\n{:s}'.format(self.mean.__str__(), self.cov.__str__())


class Lambda:

    def __init__(self, P, Q):
        self.P = P
        self.Q = Q
        self.init_rnd()

    def init_rnd(self):
        self.mean = np.random.normal(loc=0.0, scale=0.1, size=self.P*self.Q).reshape(self.P, self.Q)
        self.pre = np.empty((self.P, self.Q, self.Q))
        self.cov = np.empty((self.P, self.Q, self.Q))
        for p in range(self.P):
            self.pre[p] = np.eye(self.Q)
            if self.Q > 0:
                self.cov[p] = np.linalg.inv(self.pre[p])
            else:
                self.cov[p] = self.pre[p]

    def __str__(self, cov=True):
        s = 'mean:\n{:s}'.format(self.mean.__str__())
        if cov:
            for p in range(self.P):
                s += '\ncov[{:d}]:\n{:s}'.format(p, self.cov[p].__str__())
        return s


class LambdaMu:

    def __init__(self, P, Q, s):
        self.P = P
        self.Q = Q
        self.s = s
        self.m = Mu(P)
        self.l = Lambda(P, Q)
        self.init_rnd()

    def init_rnd(self):
        self.m.init_rnd()
        self.l.init_rnd()
        self.pre_lm = np.zeros((self.P, self.Q))
        self.build_cov()

    def update(self, h, q_nu, q_x, q_s, y, update_pre=True, update_mean=True):
        P = self.P
        Q = self.Q
        N = q_s.s.shape[1]
        if update_pre:
            # l.pre
            assert np.all(q_nu.b > 0.0)
            tt = np.diag(q_nu.a/q_nu.b)
            t = np.zeros((Q, Q))
            for n in range(N):
                t += q_s.s[self.s, n]*(np.outer(q_x.mean[:, n], q_x.mean[:, n])+q_x.cov)
            self.l.pre = np.empty((P, Q, Q))
            for p in range(P):
                self.l.pre[p] = tt+h.psii_d[p]*t
                assert is_pd(self.l.pre[p])
            # m.pre
            self.m.pre = h.nu+h.psii_d*np.sum(q_s.s[self.s,:])
            assert np.all(self.m.pre > 0.0)
            # pre_lm
            self.pre_lm = np.outer(h.psii_d, q_x.mean.dot(q_s.s[self.s, :]))
            self.build_cov()

        if update_mean:
            # l.mean
            self.l.mean = np.empty((P, Q))
            for p in range(P):
                w = np.multiply(q_s.s[self.s, :], y[p])
                self.l.mean[p] = self.cov[p, :Q, :Q].dot(h.psii_d[p]*q_x.mean.dot(w))

            # m.mean
            self.m.mean = np.multiply(h.psii_d, y.dot(q_s.s[self.s, :]))+np.multiply(h.mu, h.nu)
            self.m.mean = np.multiply(self.cov[:, Q, Q], self.m.mean)

    def build_cov(self):
        P = self.P
        Q = self.Q
        self.pre = np.empty((P, Q+1, Q+1))
        self.cov = np.empty((P, Q+1, Q+1))
        self.pre[:, Q, Q] = self.m.pre
        self.cov_lm = np.empty((P, Q))
        for p in range(P):
            self.pre[p, :Q, :Q] = self.l.pre[p]
            self.pre[p, :Q, Q] = self.pre_lm[p]
            self.pre[p, Q, :Q] = self.pre_lm[p]
            assert is_pd(self.pre[p])
            self.cov[p] = np.linalg.inv(self.pre[p])
            assert is_pd(self.cov[p])
            self.l.cov[p] = self.cov[p, :Q, :Q]
            self.cov_lm[p] = self.cov[p, :Q, Q]
        self.m.cov = self.cov[:, Q, Q]
        assert(np.all(self.m.cov > 0))

    def __str__(self):
        s = 'mu:\n{:s}\n\nlambda:\n{:s}'.format(self.m.__str__(), self.l.__str__(cov=False))
        for p in range(self.P):
            s += '\n\ncov[{:d}]:\n{:s}'.format(p, self.cov[p].__str__())
        return s


class X:

    def __init__(self, Q, N):
        self.Q = Q
        self.N = N
        self.init_rnd()

    def init_rnd(self):
        self.mean = np.random.normal(loc=0.0, scale=1e-1, size=self.Q*self.N).reshape(self.Q, self.N)
        self.cov = np.diag(np.random.normal(loc=1.0, scale=1e-5, size=self.Q))
        if self.Q > 0:
            self.pre = np.linalg.inv(self.cov)
        else:
            self.pre = self.cov

    def update(self, h, q_lm, y, update_pre=True):
        if update_pre:
            self.pre = np.eye(self.Q) + np.transpose(q_lm.l.mean).dot(h.psii).dot(q_lm.l.mean)
            for i in range(self.Q):
                for j in range(self.Q):
                    self.pre[i, j] += h.psii_d.dot(q_lm.l.cov[:, i, j])
            assert is_pd(self.pre)
            if self.Q > 0:
                self.cov = np.linalg.inv(self.pre)
            else:
                self.cov = self.pre

        B = self.cov.dot(np.transpose(q_lm.l.mean)).dot(h.psii)
        v = np.transpose(q_lm.cov_lm).dot(h.psii_d)
        #a = -B.dot(q_lm.m.mean)-self.cov.dot(v)
        a = -B.dot(q_lm.m.mean)
        self.mean = B.dot(y)+a[:, np.newaxis]

         # vv = np.zeros(self.Q)
         # for q in range(self.Q):
         #     for p in range(len(h.psii_d)):
         #         vv[q] += h.psii_d[p] * q_lm.cov_lm[p, q]
         # mean = np.zeros((self.Q, self.N))
         # vv = np.zeros(self.Q)
         # for n in range(self.N):
         #     mean[:,n] = self.cov.dot(np.transpose(q_lm.l.mean).dot(h.psii).dot((y[:,n]-q_lm.m.mean)[:, np.newaxis]) - vv[:, np.newaxis]).squeeze()
         # self.mean = mean

    def __str__(self):
        return 'mean^T:\n{:s}\ncov:\n{:s}'.format(np.transpose(self.mean).__str__(), self.cov.__str__())


class S:

    def __init__(self, S, N):
        self.S = S
        self.N = N
        self.init_rnd()

    def init_rnd(self):
        self.s = np.random.gamma(shape=5.0, scale=0.01, size=self.S*self.N).reshape(self.S, self.N)
        self.normalize()

    def normalize(self):
        self.s /= np.sum(self.s, 0)

    def update(self, h, y, s, q_pi, q_lm, q_x):
        P = len(y)
        Q = h.Q
        N = self.N
        lm_cov_d = np.empty((P, Q+1))
        for p in range(P):
            lm_cov_d[p] = np.diagonal(q_lm.cov[p])
        xt_mean_s = np.vstack((q_x.mean**2, np.ones((1, N))))
        xt_cov_d = np.hstack((np.diagonal(q_x.cov), [0]))
        lm_mean_s = np.hstack((q_lm.l.mean, q_lm.m.mean.reshape(P, 1)))**2

        # n independent
        if Q > 0:
            const = digamma(q_pi.alpha[s])+0.5*np.log(np.abs(np.linalg.det(q_x.cov)))
        else:
            const = digamma(q_pi.alpha[s])
        # 1
        const -= 0.5*h.psii_d.dot((lm_mean_s+lm_cov_d).dot(xt_cov_d))

        # trace
        log_qs = np.zeros(N)
        yd = y-q_lm.l.mean.dot(q_x.mean)-q_lm.m.mean.reshape((P, 1))
        for n in range(N):
            log_qs[n] = -0.5*np.trace(h.psii.dot(np.outer(yd[:,n], yd[:,n])))
        # 2
        log_qs -= 0.5*h.psii_d.dot(lm_cov_d.dot(xt_mean_s))
        log_qs += const
        self.s[s, :] = log_qs

    def __str__(self):
        return np.transpose(self.s).__str__()
