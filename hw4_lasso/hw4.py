from numpy import array, abs, delete, copy
from numpy.linalg import norm
from numpy.random import normal, shuffle
from numba import jit


class LASSORegression:
    def __init__(self, lamb, X, Y):
        self.lamb = lamb
        self.X = X.T
        self.Y = array(Y)
        self.n, self.d = X.shape
        self.betas = normal(loc=0.00001, scale=0.0000001, size=self.d)  # init betas to small nonzero numbers
        self.x_without_j_lookup = {}
        self.x_j_lookup = {}
        for j in range(self.d):  # cache values that won't change over the course of coordinate descent
            self.x_without_j_lookup[j] = delete(self.X, j, axis=0)  # remove the jth column
            self.x_j_lookup[j] = self.X[j]

    @jit
    def _objective(self):
        """The objective function"""
        return 1/self.n * norm(self.Y - (self.X.T @ self.betas))**2 + self.lamb * abs(self.betas).sum()

    @jit
    def _partial_min_solution(self, j):
        """Solution to the partial minimization function"""
        beta_without_j = delete(self.betas, j, axis=0)
        X_without_j = self.x_without_j_lookup[j]
        X_j = self.x_j_lookup[j]
        R_without_j = (self.Y - (beta_without_j.T @ X_without_j))
        c_j = 2/self.n * (X_j @ R_without_j)
        a_j = 2 * sum(X_j**2)
        if abs(c_j) <= self.lamb:
            return 0
        if c_j < -self.lamb:
            return (c_j + self.lamb) / (a_j / self.n)
        elif c_j > self.lamb:
            return (c_j - self.lamb) / (a_j / self.n)

    def cyclic_coord_descent(self, max_cycles=10, verbose=False, optimize=False):
        self.betas = normal(loc=0.00001, scale=0.0000001, size=self.d)  # init betas to small nonzero numbers
        beta_history = []
        objective_history = [] if optimize else [self._objective()]
        for cycle in range(max_cycles):
            if verbose:
                print("Starting cycle {}".format(cycle+1))
            for j in range(self.d):
                self.betas[j] = self._partial_min_solution(j)
            if not optimize:
                objective_history.append(self._objective())
                beta_history.append(copy(self.betas))
        if verbose:
            print("Cyclic coordinate descent complete")
        return self.betas, beta_history, objective_history

    def _random_coords(self):
        """Returns the indices of the coordinates in random order"""
        coords = array(range(0, self.d))
        shuffle(coords)
        return coords

    def random_coord_descent(self, max_cycles=10, verbose=False, optimize=False):
        self.betas = normal(loc=0.00001, scale=0.0000001, size=self.d)  # init betas to small nonzero numbers
        objective_history = [] if optimize else [self._objective()]
        beta_history = []
        for cycle in range(max_cycles):
            if verbose:
                print("Starting cycle {}".format(cycle+1))
            for j in self._random_coords():
                self.betas[j] = self._partial_min_solution(j)
            if not optimize:
                objective_history.append(self._objective())
                beta_history.append(copy(self.betas))
        if verbose:
            print("Random coordinate descent complete")
        return self.betas, beta_history, objective_history

    def predict(self, newX):
        return newX.T @ self.betas
