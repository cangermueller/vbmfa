import numpy as np
import numpy.testing as npt
import unittest
import vbfa
import vbmfa
import ipdb

class VbMfaTest(unittest.TestCase):
    def test_pi(self):
        np.random.seed(0)
        S = 10
        q_pi = vbmfa.Pi(S)
        self.assertEqual(len(q_pi.alpha), S)
        self.assertGreater(str(q_pi), 0)

    def test_s(self):
        np.random.seed(0)
        S = 10
        N = 100
        q_s = vbmfa.S((S, N))
        self.assertEqual(q_s.shape, (S, N))
        self.assertGreater(str(q_s), 0)

    def test_update(self):
        np.random.seed(0)
        P = 10
        Q = 5
        S = 4
        N = 100
        y = np.random.rand(P, N)
        mfa = vbmfa.VbMfa(vbmfa.Hyper(P, Q, S), y)
        for i in range(10):
            # print "Interation {:d}".format(i)
            # fas
            mse = mfa.mse()
            # print mse
            mfa.update_fas()
            self.assertLess(mfa.mse() - mse, 1e-5 + 1.0)    # TODO: depends on initialization
            # s
            mse = mfa.mse()
            # print mse
            mfa.update_s()
            self.assertLess(mfa.mse() - mse, 1e-5 + 1.0)    # TODO: depends on initialization
            npt.assert_allclose(np.sum(mfa.q_s, 0), 1.0)
            # pi
            mse = mfa.mse()
            # print mse
            mfa.update_pi()
            self.assertLess(mfa.mse() - mse, 1e-5)
            npt.assert_allclose(np.sum(mfa.q_pi.expectation()), 1.0)
            # print mfa.mse()

    def test_single_component(self):
        np.random.seed(0)
        P = 100
        Q = 50
        N = 200
        y = np.random.rand(P, N)
        # VbFa
        np.random.seed(0)
        fa = vbfa.VbFa(vbfa.Hyper(P, Q), y)
        fa.fit(maxit=5)
        # VbMfa
        np.random.seed(0)
        mfa = vbmfa.VbMfa(vbmfa.Hyper(P, Q, 1), y)
        mfa.fit(maxit=5)
        # Check equality
        self.assertEqual(fa.mse(), mfa.mse())
        npt.assert_array_equal(fa.q_mu.mean, mfa.fas[0].q_mu.mean)
        npt.assert_array_equal(fa.q_lambda.mean, mfa.fas[0].q_lambda.mean)

    @unittest.skip('not yet required')
    def test_map(self):
        np.random.seed(0)
        P = 10
        Q = 5
        S = 2
        N = 3
        Y = np.random.rand(P, N)
        mfa = vbmfa.VbMfa(vbmfa.Hyper(P, Q, S), Y)
        mfa.q_s.fill(0.0)
        mfa.q_s[0, 0] = 1.0
        mfa.q_s[1, 1] = 1.0
        mfa.q_s[0, 2] = 0.5
        mfa.q_s[1, 2] = 0.5
        map_x = mfa.map_x()
        npt.assert_array_equal(map_x[:, 0], mfa.fas[0].q_x.mean[:, 0])



if __name__ == '__main__':
    unittest.main()


