from unittest import TestCase
from .hw3 import RidgeRegression
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from numpy import array
from sklearn.linear_model import LogisticRegression

class TestRidgeRegression(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = read_csv("hw3_fastgrad_ridge/spam.csv")
        last_col = cls.test_data.columns[-1]
        cls.Y = (cls.test_data[last_col] * 2)-1
        spam = cls.test_data.drop(last_col, axis=1)
        cls.stdscaler = StandardScaler().fit(spam)
        cls.X = cls.stdscaler.transform(spam)
        cls.X_simple = StandardScaler().fit(spam[spam.columns[0:2]]).transform(spam[spam.columns[0:2]])

    def test_objective(self):
        r = RidgeRegression(0.1, self.X_simple, self.Y)
        n, p = self.X.shape
        self.assertAlmostEqual(
            0.69314718055994529,
            r._objective(array([0.0, 0.0]))
        )
        self.assertAlmostEqual(
            r._objective(array([0.0, 0.0])),
            r._objective_long_way(array([0.0, 0.0]))
        )

    def test_gradient(self):
        r = RidgeRegression(0.1, self.X_simple, self.Y)
        grad = list(r._grad(array([0.0, 0.0])))
        self.assertEqual(2, len(grad))
        self.assertAlmostEqual(
            -0.061670642732909037,
            grad[0]
        )
        self.assertAlmostEqual(
            0.0147685870655,
            grad[1]
        )
        grad = r._grad(array([0.0, 0.0]))
        grad_long = r._grad_long_way(array([0.0, 0.0]))
        for p in range(2):
            self.assertAlmostEqual(grad_long[p], grad[p])

    # def test_fast_grad(self):
    #     #r = RidgeRegression(0.1, self.X_simple, self.Y)
    #     r = RidgeRegression(0.1, self.X, self.Y)
    #     #r.estimate_init_stepsize()
    #     beta, beta_hist, theta_hist, obj_hist = r.do_fastgrad(1, 0.001)
    #     print(beta)
    #     sklog = LogisticRegression(penalty='l2', C=1/0.1, fit_intercept=False).fit(self.X, self.Y)
    #     self.assertAlmostEqual(sklog.coef_[0][0], beta[0])
    #     self.assertAlmostEqual(sklog.coef_[0][1], beta[1])

    def test_grad_descent(self):
        r = RidgeRegression(0.1, self.X_simple, self.Y)
        #r = RidgeRegression(0.1, self.X, self.Y)
        beta, beta_hist, obj_hist = r.do_grad_descent(1, 0.001)
        sklog = LogisticRegression(penalty='l2', C=1/0.1, fit_intercept=False).fit(self.X_simple, self.Y)
        self.assertAlmostEqual(sklog.coef_[0][0], beta[0])
        self.assertAlmostEqual(sklog.coef_[0][1], beta[1])
