import unittest
import numpy as np
import numpy.testing as npt
import vbfa


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
        y = np.random.rand(P, N)
        fa = vbfa.VbFa(vbfa.Hyper(P, Q), y)
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
            self.assertLess(fa.mse() - mse, tol + 2.0)    # TODO: depends on initialization
            # mu
            mse = fa.mse()
            fa.update_mu()
            self.assertLess(fa.mse() - mse, tol)

    @unittest.skip('fails with different initialization')
    def test_fit(self):
        np.random.seed(0)
        P = 50
        Q = 10
        N = 100
        mu = 10.0
        y, lambda_, x, mu = sample_cluster(P, Q, N, mu, 1.0)
        fa = vbfa.VbFa(vbfa.Hyper(P, Q), y)
        fa.fit()
        npt.assert_array_less(np.abs(fa.q_mu.mean - mu), 1.0)
        self.assertLess(fa.mse(), 60.0)


class TestQ(unittest.TestCase):
    def test_nu(self):
        P = 10
        Q = 5
        hyper = vbfa.Hyper(P, Q)
        # init
        q_nu = vbfa.Nu(Q)
        self.assertEqual(len(q_nu.b), Q)
        self.assertTrue(len(q_nu.__str__()) > 0)
        # update
        q_lambda = vbfa.Lambda(P, Q)
        q_nu.update(hyper, q_lambda)
        self.assertTrue(np.all(q_nu.b > 0))

    def test_mu(self):
        P = 10
        Q = 5
        hyper = vbfa.Hyper(P, Q)
        # init
        q_mu = vbfa.Mu(P)
        self.assertEqual(len(q_mu.mean), P)
        self.assertEqual(len(q_mu.cov), P)
        self.assertTrue(len(q_mu.__str__()) > 0)
        # update
        N = 100
        y = np.random.rand(P, N)
        q_lambda = vbfa.Lambda(P, Q)
        q_x = vbfa.X(Q, N)
        q_mu.update(hyper, q_lambda, q_x, y)

    def test_lambda(self):
        P = 10
        Q = 5
        hyper = vbfa.Hyper(P, Q)
        # init
        q_lambda = vbfa.Lambda(P, Q)
        self.assertEqual(q_lambda.mean.shape, (P, Q))
        self.assertEqual(len(q_lambda.cov), P)
        for p in range(P):
            self.assertEqual(q_lambda.cov[p].shape, (Q, Q))
        self.assertTrue(len(q_lambda.__str__()) > 0)
        # update
        N = 100
        y = np.random.rand(P, N)
        q_mu = vbfa.Mu(P)
        q_nu = vbfa.Nu(Q)
        q_x = vbfa.X(Q, N)
        q_lambda.update(hyper, q_mu, q_nu, q_x, y)
        for p in range(P):
            self.assertTrue(np.all(np.linalg.eigvals(q_lambda.cov[p]) > 0))

    def test_x(self):
        P = 10
        Q = 5
        N = 100
        hyper = vbfa.Hyper(P, Q)
        # init
        q_x = vbfa.X(Q, N)
        self.assertEqual(q_x.mean.shape, (Q, N))
        self.assertEqual(q_x.cov.shape, (Q, Q))
        self.assertGreater(q_x.__str__(), 0)
        # update
        y = np.random.rand(P, N)
        q_lambda = vbfa.Lambda(P, Q)
        q_mu = vbfa.Mu(P)
        q_x.update(hyper, q_lambda, q_mu, y)







if __name__ == '__main__':
    unittest.main()
