import numpy as np
import copy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pdb

import q as qdist


class Hyper:
    def __init__(self, P, Q=None, S=1):
        self.P = P
        self.Q = P if Q is None else Q
        self.S = S
        self.init()

    def init(self):
        self.alpha = 1.0
        self.m = np.empty(self.S)
        self.m.fill(self.S**-1)
        self.a = 1.0
        self.b = 1.0
        self.mu = np.zeros(self.P)
        self.nu = np.ones(self.P)
        self.psi_d = np.ones(self.P)
        self.init_psi()

    def init_psi(self):
        self.psii_d = self.psi_d**-1
        self.psi = np.diag(self.psi_d)
        self.psii = np.diag(self.psii_d)

    def __str__(self):
        st = ''
        st += 'alpha: {:s}'.format(self.alpha.__str__())
        st += '\na: {:f}, b: {:f}'.format(self.a, self.b)
        st += '\nmu: {:s}'.format(self.mu.__str__())
        st += '\nnu: {:s}'.format(self.nu.__str__())
        st += '\npsi: {:s}'.format(self.psi_d.__str__())
        return st


class Model(object):

    def __init__(self, h, y):
        self.y = y
        self.h = h
        self.P = h.P
        self.Q = h.Q
        self.S = h.S
        if (isinstance(y, int)):
            self.N = y
        else:
            self.N = y.shape[1]
        self.q_pi = qdist.Pi(self.S)
        self.q_nu = [qdist.Nu(self.P, self.Q) for s in range(self.S)]
        self.q_lm = [qdist.LambdaMu(self.P, self.Q, s) for s in range(self.S)]
        self.q_m = [self.q_lm[s].m for s in range(self.S)]
        self.q_l = [self.q_lm[s].l for s in range(self.S)]
        self.q_x = [qdist.X(self.Q, self.N) for s in range(self.S)]
        self.q_s = qdist.S(self.S, self.N)
        self.init_rnd()

    def init_rnd(self):
        self.q_pi.init_rnd()
        for s in range(self.S):
            self.q_nu[s].init_rnd()
            self.q_lm[s].init_rnd()
            self.q_x[s].init_rnd()
        self.q_s.init_rnd()

    def init_complex(self, hard=True):
        km = KMeans(self.S)
        self.init_rnd()
        y = np.transpose(self.y)
        labels = km.fit_predict(y)
        centroids = km.cluster_centers_
        for s in range(self.S):
            self.q_lm[s].m.mean = centroids[s]
        if hard == True:
            self.q_s.s = np.random.normal(1.0, 1e-3, size=self.q_s.s.size).reshape((self.S, -1))
            for n in range(self.N):
                self.q_s.s[labels[n], n] = 100
            self.q_s.normalize()
        else:
            for s in range(self.S):
                self.q_s.s[s] = np.linalg.norm(y - centroids[s], axis=1)
            self.q_s.s = np.maximum(1 - self.q_s.s / np.max(self.q_s.s, axis=0), 0.01)
            self.q_s.normalize()

        pca = PCA(self.Q)
        for s in range(self.S):
            y = np.transpose(self.y[:, labels == s])
            x = pca.fit_transform(y)
            self.q_lm[s].l.mean = np.transpose(pca.components_)
            self.q_x[s].mean[:, labels == s] = np.transpose(x)

        self.update_nu()
        #self.update_s()
        self.update_pi()

    def infer(self, maxit=10, eps=0.01, times=3, update=None):
        models = [copy.deepcopy(self)]
        mses = [self.mse()]
        c = 0
        if update is None:
            update = self.update
        for i in range(maxit):
            update(self, i)
            models.append(copy.deepcopy(self))
            mses.append(self.mse())
            if (mses[-2]-mses[-1]) <= eps:
                c += 1
            else:
                c = 0
            if c == times:
                break
        return [mses, models]

    def update(self, model, it):
        model.update_x()
        model.update_lm()
        model.update_s()
        model.update_pi()
        model.update_nu()

    def update_pi(self):
        self.q_pi.update(self.h, self.q_s)

    def update_nu(self):
        for s in range(self.S):
            self.q_nu[s].update(self.h, self.q_l[s])

    def update_m(self):
        for s in range(self.S):
            self.q_m[s].update(self.h, self.y, self.q_l[s], self.q_s, self.q_x[s])
            self.q_lm[s].build_cov()

    def update_l(self):
        for s in range(self.S):
            self.q_l[s].update(self.h, self.y, self.q_m[s], self.q_nu[s], self.q_s, self.q_x[s])
            self.q_lm[s].build_cov()

    def update_lm(self, **kwargs):
        for s in range(self.S):
            self.q_lm[s].update(self.h, self.y, self.q_nu[s], self.q_s, self.q_x[s], **kwargs)

    def update_x(self):
        for s in range(self.S):
            self.q_x[s].update(self.h, self.q_lm[s], self.y)

    def update_s(self):
        for s in range(self.S):
            self.q_s.update(self.h, self.y, s, self.q_pi, self.q_lm[s], self.q_x[s])
        self.q_s.normalize()


    def __str__(self):
        st = 'q_pi:\n{:s}'.format(self.q_pi.__str__())
        for s in range(self.S):
            st += '\n\nq_nu[{:d}]:\n{:s}'.format(s, self.q_nu[s].__str__())
        for s in range(self.S):
            st += '\n\nq_lm[{:d}]:\n{:s}'.format(s, self.q_lm[s].__str__())
        for s in range(self.S):
            st += '\n\nq_x[{:d}]:\n{:s}'.format(s, self.q_x[s].__str__())
        st += '\n\nq_s:\n{:s}'.format(self.q_s.__str__())
        return st

    def predict_y_s(self, s):
        return self.q_l[s].mean.dot(self.q_x[s].mean)+self.q_m[s].mean[:, np.newaxis]

    def predict_y(self, ss=None):
        if ss is None:
            ss = range(self.S)
        q_s = self.q_s.s[ss]
        q_s /= np.sum(q_s, 0)
        yp = np.zeros((self.P, self.N))
        for i in range(len(ss)):
            yp += self.predict_y_s(ss[i]).dot(np.diag(q_s[i]))
        return yp

    def sample_y(self, ss=None):
        y_mean = self.predict_y(ss)
        y = np.empty((self.P, self.N))
        for n in range(self.N):
            y[:, n] = np.random.multivariate_normal(mean=y_mean[:, n], cov=self.h.psi, size=1)[:, np.newaxis]
        return y

    def mse(self):
        dy = (self.y-self.predict_y()).ravel('F')
        return dy.dot(dy)/self.y.shape[1]

