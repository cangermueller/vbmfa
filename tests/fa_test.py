"""Test cases for fa.py"""

import unittest
import numpy as np
import numpy.testing as npt
import vbmfa.fa as vbfa


def sample_cluster(P=10, Q=5, N=100, mu=0.0, sigma=1.0):
    lambda_ = np.random.normal(0.0, 1.0, P * Q).reshape(P, Q)
    mu = np.random.normal(mu, 1.0, P)
    x = np.random.normal(0.0, 1.0, Q * N).reshape(Q, N)
    y = lambda_.dot(x) + mu[:, np.newaxis] + np.random.normal(0.0, sigma, P * N).reshape(P, N)
    return (y, lambda_, x, mu)


class TestVbFa(unittest.TestCase):

    def test_update(self):
        np.random.seed(0)
        P = 30
        Q = 10
        N = 100
        Y = np.random.rand(P, N)
        fa = vbfa.VbFa(Y, Q)
        tol = 1e-5
        for i in range(10):
            # nu
            mse = fa.mse()
            fa.update_nu()
            self.assertLess(fa.mse() - mse, tol)
            # lambda
            mse = fa.mse()
            fa.update_lambda()
            self.assertLess(fa.mse() - mse, tol)
            # x
            mse = fa.mse()
            fa.update_x()
            self.assertLess(fa.mse() - mse, tol + 1.5) # TODO: depends on initialization
            # mu
            mse = fa.mse()
            fa.update_mu()
            self.assertLess(fa.mse() - mse, tol)

    def test_fit(self):
        np.random.seed(0)
        P = 50
        Q = 10
        N = 100
        mu = 10.0
        Y, lambda_, x, mu = sample_cluster(P, Q, N, mu, 1.0)
        fa = vbfa.VbFa(Y, Q)
        fa.init()
        for i in range(10):
            fa.update()
        npt.assert_array_less(np.abs(fa.q_mu.mean - mu), 1.0)
        self.assertLess(fa.mse(), 60.0)

    def test_permute(self):
        np.random.seed(0)
        P = 30
        Q = 10
        N = 100
        Y = np.random.rand(P, N)
        fa = vbfa.VbFa(Y, Q)
        mse = fa.mse()
        order = np.arange(Q)
        np.random.shuffle(order)
        fa.permute(order)
        self.assertAlmostEqual(fa.mse(), mse)

class TestNu(unittest.TestCase):
    def test_init(self):
        P = 10
        Q = 5
        q_nu = vbfa.Nu(Q)
        self.assertEqual(len(q_nu.b), Q)
        self.assertTrue(len(q_nu.__str__()) > 0)

    def test_update(self):
        P = 10
        Q = 5
        hyper = vbfa.Hyper(P, Q)
        q_lambda = vbfa.Lambda(P, Q)
        q_nu = vbfa.Nu(Q)
        q_nu.update(hyper, q_lambda)
        self.assertTrue(np.all(q_nu.b > 0))


class TestMu(unittest.TestCase):
    def test_init(self):
        P = 10
        Q = 5
        q_mu = vbfa.Mu(P)
        self.assertEqual(len(q_mu.mean), P)
        self.assertEqual(len(q_mu.cov), P)
        self.assertTrue(len(q_mu.__str__()) > 0)

    def test_update(self):
        np.random.seed(0)
        P = 10
        Q = 5
        N = 100
        q_mu = vbfa.Mu(P)
        y = np.random.rand(P, N)
        hyper = vbfa.Hyper(P, Q)
        q_lambda = vbfa.Lambda(P, Q)
        q_x = vbfa.X(Q, N)
        q_mu.update(hyper, q_lambda, q_x, y)


class TestLambda(unittest.TestCase):
    def test_init(self):
        P = 10
        Q = 5
        q_lambda = vbfa.Lambda(P, Q)
        self.assertEqual(q_lambda.mean.shape, (P, Q))
        self.assertEqual(len(q_lambda.cov), P)
        for p in range(P):
            self.assertEqual(q_lambda.cov[p].shape, (Q, Q))
        self.assertTrue(len(q_lambda.__str__()) > 0)

    def test_update(self):
        np.random.seed(0)
        P = 10
        Q = 5
        hyper = vbfa.Hyper(P, Q)
        q_lambda = vbfa.Lambda(P, Q)
        N = 100
        y = np.random.rand(P, N)
        q_mu = vbfa.Mu(P)
        q_nu = vbfa.Nu(Q)
        q_x = vbfa.X(Q, N)
        q_lambda.update(hyper, q_mu, q_nu, q_x, y)
        for p in range(P):
            self.assertTrue(np.all(np.linalg.eigvals(q_lambda.cov[p]) > 0))

    def test_permute(self):
        P = 10
        Q = 4
        q_lambda = vbfa.Lambda(P, Q)
        mean_before = q_lambda.mean.copy()
        cov_before = q_lambda.cov.copy()
        order = [2, 3, 1, 0]
        q_lambda.permute(order)
        self.assertEqual(q_lambda.mean.shape, (P, Q))
        for q in range(Q):
            npt.assert_equal(q_lambda.mean[:, q], mean_before[:, order[q]])
        for p in range(P):
            for i in range(Q):
                for j in range(Q):
                    self.assertEqual(q_lambda.cov[p, q, q], cov_before[p, order[q], order[q]])


class TestX(unittest.TestCase):
    def test_init(self):
        P = 10
        Q = 5
        N = 100
        q_x = vbfa.X(Q, N)
        self.assertEqual(q_x.mean.shape, (Q, N))
        self.assertEqual(q_x.cov.shape, (Q, Q))
        self.assertGreater(len(q_x.__str__()), 0)

    def test_update(self):
        np.random.seed(0)
        P = 10
        Q = 5
        N = 100
        q_x = vbfa.X(Q, N)
        hyper = vbfa.Hyper(P, Q)
        y = np.random.rand(P, N)
        q_lambda = vbfa.Lambda(P, Q)
        q_mu = vbfa.Mu(P)
        q_x.update(hyper, q_lambda, q_mu, y)




if __name__ == '__main__':
    unittest.main()
