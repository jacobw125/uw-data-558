from unittest import TestCase
from .smoothed_hinge import SmoothedHingeLossClassifier
from .svm import SVM
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from numpy import array


class TestSmoothedHingeSVM(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data = read_csv("hw3_fastgrad_logistic/spam.csv")
        last_col = cls.test_data.columns[-1]
        cls.Y = (cls.test_data[last_col] * 2)-1
        spam = cls.test_data.drop(last_col, axis=1)
        cls.stdscaler = StandardScaler().fit(spam)
        cls.X = cls.stdscaler.transform(spam)
        cls.X_simple = StandardScaler().fit(spam[spam.columns[0:2]]).transform(spam[spam.columns[0:2]])

    def test_objective(self):
        r = SmoothedHingeLossClassifier(0.1, self.X, self.Y)
        obj = r._obj(array([1.0]*r.d))
        print(obj)

    def test_gradient(self):
        r = SmoothedHingeLossClassifier(0.1, self.X_simple, self.Y)
        grad = r._grad(array([1.0]*r.d))
        print(grad)

    # def test_fast_grad(self):
    #     r = SmoothedHingeLossClassifier(0.1, self.X, self.Y)
    #     beta, beta_hist, theta_hist, obj_hist = r.train(0.001, optimize=False)
    #     print(beta)
