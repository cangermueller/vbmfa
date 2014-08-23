"""Test cases for vbmfa.py"""

import numpy as np
import numpy.testing as npt
import unittest
import vbfa
import vbmfa
import ipdb


class VbMfaTest(unittest.TestCase):
    def test_update(self):
        np.random.seed(0)
        P = 10
        Q = 5
        S = 4
        N = 100
        Y = np.random.rand(P, N)
        mfa = vbmfa.VbMfa(Y, Q, S)
        eps = 0.1
        for i in range(10):
            # fas
            mse = mfa.mse()
            mfa.update_fas()
            self.assertLess(mfa.mse() - mse, eps)
            # s
            mse = mfa.mse()
            mfa.update_s()
            self.assertLess(mfa.mse() - mse, eps)
            npt.assert_allclose(np.sum(mfa.q_s, 0), 1.0)
            # pi
            mse = mfa.mse()
            mfa.update_pi()
            self.assertLess(mfa.mse() - mse, eps)
            npt.assert_allclose(np.sum(mfa.q_pi.expectation()), 1.0)

    def test_single_fa(self):
        np.random.seed(0)
        P = 100
        Q = 50
        N = 200
        Y = np.random.rand(P, N)
        it = 5
        # VbFa
        np.random.seed(0)
        fa = vbfa.VbFa(Y, Q)
        for i in range(it):
            fa.update()
        # VbMfa S = 1
        np.random.seed(0)
        mfa = vbmfa.VbMfa(Y, Q, 1)
        for i in range(it):
            mfa.update()
        self.assertEqual(mfa.mse(), fa.mse())
        npt.assert_array_equal(mfa.fas[0].q_mu.mean, fa.q_mu.mean)
        npt.assert_array_equal(mfa.fas[0].q_lambda.mean, fa.q_lambda.mean)
        # VbMfa S = 3
        np.random.seed(0)
        mfa = vbmfa.VbMfa(Y, Q, 3)
        mfa.q_s.fill(0.0)
        mfa.q_s[0, :] = 1.0
        for i in range(it):
            mfa.update_fas()
        self.assertEqual(mfa.mse(), fa.mse())
        npt.assert_array_equal(mfa.fas[0].q_mu.mean, fa.q_mu.mean)
        npt.assert_array_equal(mfa.fas[0].q_lambda.mean, fa.q_lambda.mean)


class TestPi(unittest.TestCase):
    def test_pi(self):
        np.random.seed(0)
        S = 10
        q_pi = vbmfa.Pi(S)
        self.assertEqual(len(q_pi.alpha), S)
        self.assertGreater(len(str(q_pi)), 0)


class TestS(unittest.TestCase):
    def test_s(self):
        np.random.seed(0)
        S = 10
        N = 100
        q_s = vbmfa.S((S, N))
        self.assertEqual(q_s.shape, (S, N))
        self.assertGreater(len(str(q_s)), 0)




if __name__ == '__main__':
    unittest.main()


