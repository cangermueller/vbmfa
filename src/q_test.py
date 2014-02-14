import unittest
import numpy as np
import q as qdist


class PiTest(unittest.TestCase):
    def test_init(self):
        pi = qdist.Pi(10)
        self.assertTrue(len(pi.alpha) == 10)
        self.assertTrue((pi.alpha > 0.0).all())


class NuTest(unittest.TestCase):
    def test_init(self):
        nu = qdist.Nu(5, 10)
        self.assertTrue(len(nu.b) == 10)
        self.assertTrue((nu.b > 0.0).all())


class MuTest(unittest.TestCase):
    def test_init(self):
        P = 10
        mu = qdist.Mu(P)
        self.assertEquals(len(mu.mean), P)
        self.assertEquals(len(mu.pre), P)


class LambdaTest(unittest.TestCase):
    def test_init(self):
        P = 5
        Q = 10
        l = qdist.Lambda(P, Q)
        self.assertEqual(l.mean.shape, (P, Q))
        self.assertEqual(l.cov.shape, (P, Q, Q))
        self.assertTrue(l.cov.min > 0.0)
        for p in range(P):
            self.assertNotEqual(np.linalg.det(l.cov[p]), 0.0)
            self.assertEqual(l.pre.shape, (P, Q, Q))


class LambdaMuTest(unittest.TestCase):

    def test_init(self):
        P = 10
        Q = 5
        s = 1
        lm = qdist.LambdaMu(P, Q, s)
        self.assertEqual(lm.m.P, P)
        self.assertEqual(lm.l.P, P)
        self.assertEqual(lm.l.Q, Q)
        self.assertEqual(lm.pre_lm.shape, (P, Q))
        self.assertEqual(lm.pre.shape, (P, Q+1, Q+1))
        self.assertEqual(lm.cov.shape, (P, Q+1, Q+1))
        self.assertTrue(lm.pre.min() >= 0.0)
        # self.assertTrue(lm.cov.min() >= 0.0)
        for p in range(P):
            self.assertEqual(lm.pre[p, Q, Q], lm.m.pre[p])
            for qq in range(Q):
                self.assertEqual(lm.pre[p, qq, Q], lm.pre[p, Q, qq])
            self.assertTrue(np.linalg.eigvals(lm.pre[p]).min > 0)


class S(unittest.TestCase):

    def test_init(self):
        S = 5
        N = 100
        s = qdist.S(S, N)
        self.assertEqual(s.s.shape, (S, N))
        self.assertTrue(np.allclose(np.sum(s.s, 0), np.ones(N)))


class X(unittest.TestCase):

    def test_init(self):
        Q = 5
        N = 10
        x = qdist.X(Q, N)
        self.assertEqual(x.mean.shape, (Q, N))
        self.assertEqual(x.cov.shape, (Q, Q))
        self.assertTrue(x.cov.min > 0.0)


if __name__ == '__main__':
    unittest.main()
