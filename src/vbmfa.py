import numpy as np
import scipy.io as io
import model

def from_q_lm(q_lm):
    P = q_lm.P
    Q = q_lm.Q
    QQ = Q + 1
    Lm = np.hstack((q_lm.m.mean[:, np.newaxis], q_lm.l.mean))
    Lcov = np.zeros((QQ, QQ, P))
    for p in range(P):
        Lcov[0, 0, p] = q_lm.cov[p, Q, Q]
        Lcov[1:QQ, 0, p] = q_lm.cov[p, :Q, Q]
        Lcov[0, 1:QQ, p] = q_lm.cov[p, Q, :Q]
        Lcov[1:QQ, 1:QQ, p] = q_lm.cov[p, :Q, :Q]
    return [Lm, Lcov]

def from_q_x(q_x):
    Q = q_x.Q
    QQ = Q+1
    Xm = np.ones((QQ, q_x.N))
    Xm[1:QQ] = q_x.mean
    Xcov = np.zeros((QQ, QQ))
    Xcov[1:QQ, 1:QQ] = q_x.cov
    return [Xm, Xcov]

def from_q_s(q_s):
    return np.transpose(q_s.s)

def from_q_nu(q_nu):
    return [q_nu.a, q_nu.b]

def from_q_pi(q_pi):
    return q_pi.alpha

def from_model(m):
    model = dict()
    model['Y'] = m.y

    model['alpha'] = np.sum(m.h.alpha * m.h.m)
    model['pa'] = m.h.a
    model['pb'] = m.h.b
    model['mean_mcl'] = m.h.mu[:, np.newaxis]
    model['nu_mcl'] = m.h.nu[:, np.newaxis]
    model['psii'] = m.h.psii_d[:, np.newaxis]

    model['u'] = from_q_pi(m.q_pi)
    model['Qns'] = from_q_s(m.q_s)
    model['Lm'] = np.empty(m.S, dtype=np.object)
    model['Lcov'] = np.empty(m.S, dtype=np.object)
    model['Xm'] = np.empty(m.S, dtype=np.object)
    model['Xcov'] = np.empty(m.S, dtype=np.object)
    model['b'] = np.empty(m.S, dtype=np.object)
    for s in range(m.S):
        Lm, Lcov = from_q_lm(m.q_lm[s])
        model['Lm'][s] = Lm
        model['Lcov'][s] = Lcov
        Xm, Xcov = from_q_x(m.q_x[s])
        model['Xm'][s] = Xm
        model['Xcov'][s] = Xcov
        a, b = from_q_nu(m.q_nu[s])
        model['a'] = a
        model['b'][s] = b

    model['removal'] = 0
    return model
