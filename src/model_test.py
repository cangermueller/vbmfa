import unittest
import numpy as np
import model
import pdb


class TestHyper(unittest.TestCase):

    def test_init(self):
        P = 10
        Q = 5
        S = 3
        h = model.Hyper(P, Q, S)
        self.assertEqual(h.psi.shape, (P, P))
        self.assertEqual(h.m.shape, (S, ))
        self.assertEqual(h.mu.shape, (P, ))
        self.assertEqual(h.nu.shape, (P, ))
        self.assertTrue(h.nu.min > 0.0)


class TestModel(unittest.TestCase):

    def test_init(self):
        P = 10
        Q = 5
        S = 3
        N = 100
        y = np.random.rand(P, N)
        h = model.Hyper(P, Q, S)
        m = model.Model(h, y)
        self.assertEqual(len(m.q_nu), S)
        self.assertEqual(len(m.q_lm), S)
        self.assertEqual(len(m.q_x), S)

    def is_pd(self, m):
        return np.diagonal(m).min > 0.0 and (np.linalg.eigvals(m) > 0).all()

    def test_update_q(self):
        P = 10
        Q = 5
        S = 3
        N = 100
        y = np.random.rand(P, N)
        h = model.Hyper(P, Q, S)
        m = model.Model(h, y)

        m.q_pi.update(h, m.q_s)
        self.assertTrue((m.q_pi.alpha >= h.alpha*h.m).all())
        for s in range(S):
            m.q_nu[s].update(h, m.q_lm[s])
            self.assertEqual(m.q_nu[s].a, h.a+0.5*P)
            self.assertTrue((m.q_nu[s].b > h.b).all())
        for s in range(S):
            m.q_lm[s].update(h, m.q_nu[s], m.q_x[s], m.q_s, y)
            self.assertTrue(m.q_lm[s].m.pre.min() >= 0.0)
            for p in range(P):
                self.assertTrue(self.is_pd(m.q_lm[s].l.pre[p]))
                self.assertTrue(self.is_pd(m.q_lm[s].l.cov[p]))
                self.assertTrue(self.is_pd(m.q_lm[s].pre[p]))
                self.assertTrue(self.is_pd(m.q_lm[s].cov[p]))
        for s in range(S):
            m.q_x[s].update(h, m.q_lm[s], y)
            # print np.diagonal(m.q_x[s].cov), np.min(m.q_x[s].cov), np.max(m.q_x[s].cov)
            # print m.q_x[s].mean
        for s in range(S):
            m.q_s.update(h, y, s, m.q_pi, m.q_lm[s], m.q_x[s])
        m.q_s.s = np.maximum(1e-5, np.exp(m.q_s.s))
        m.q_s.s /= np.sum(m.q_s.s, 0)

    def test_random(self):
        P = 10
        Q = 5
        S = 3
        N = 100
        y = np.random.rand(P, N)
        h = model.Hyper(P, Q, S)
        m = model.Model(h, y)
        for it in range(10):
            m.update()






if __name__ == '__main__':
    unittest.main()
