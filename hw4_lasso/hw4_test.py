from unittest import TestCase
from .hw4 import LASSORegression
from numpy import load
from numpy import array


class TestLassoRegression(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = load("hw4_lasso/x_train.npy")
        cls.Y = load("hw4_lasso/y_train.npy")

    def test_objective(self):
        r = LASSORegression(0.1, self.X, self.Y)
        obj = r._objective()
        print(obj)

    def test_min_solution(self):
        r = LASSORegression(0.1, self.X, self.Y)
        new_b0 = r._partial_min_solution(r.d-1)
        print(new_b0)

    def test_parallel_cd(self):
        r = LASSORegression(0.1, self.X, self.Y)
        betas = r.parallel_random_coordinate_descent(max_cycles=100)
        print(betas)