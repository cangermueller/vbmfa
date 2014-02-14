import numpy as np
import q as qdist


class Hyper:
    def __init__(self, P, Q, S):
        self.P = P
        self.Q = Q
        self.S = S
        self.alpha = 1.0
        self.m = np.ones(S)
        self.a = 1.0
        self.b = 1.0
        self.mu = np.random.normal(loc=0.0, scale=1.0, size=P)
        self.nu = np.random.normal(loc=1.0, scale=1e-3, size=P)
        self.psi = np.eye(P)
        self.psii = np.linalg.inv(self.psi)
        self.psii_d = np.diagonal(self.psii)

    def __str__(self):
        st = ''
        st += 'alpha: {:s}'.format(self.alpha.__str__())
        st += '\na: {:f}, b: {:f}'.format(self.a, self.b)
        st += '\nm: {:s}'.format(self.m.__str__())
        st += '\nnu: {:s}'.format(self.nu.__str__())
        st += '\npsi:\n{:s}'.format(self.psi.__str__())
        return st


class Model(object):

    def __init__(self, h, y):
        self.y = y
        self.h = h
        self.P = h.P
        self.Q = h.Q
        self.S = h.S
        self.N = y.shape[1]
        self.q_pi = qdist.Pi(self.S)
        self.q_nu = [qdist.Nu(self.P, self.Q) for s in range(self.S)]
        self.q_lm = [qdist.LambdaMu(self.P, self.Q, s) for s in range(self.S)]
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

    def update(self):
        self.update_pi()
        self.update_nu()
        self.update_lm()
        self.update_x()
        self.update_s()

    def update_pi(self):
        self.q_pi.update(self.h, self.q_s)

    def update_nu(self):
        for s in range(self.S):
            self.q_nu[s].update(self.h, self.q_lm[s])

    def update_lm(self):
        for s in range(self.S):
            self.q_lm[s].update(self.h, self.q_nu[s], self.q_x[s], self.q_s, self.y)

    def update_x(self):
        for s in range(self.S):
            self.q_x[s].update(self.h, self.q_lm[s], self.y)

    def update_s(self):
        for s in range(self.S):
            self.q_s.update(self.h, self.y, s, self.q_pi, self.q_lm[s], self.q_x[s])
        self.q_s.s = np.exp(self.q_s.s)
# self.q_s.s = np.maximum(1e-5, np.exp(self.q_s.s))
        self.q_s.s /= np.sum(self.q_s.s, 0)


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
