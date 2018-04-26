from numpy import array, abs, delete
from numpy.linalg import norm
from numpy.random import normal, shuffle


class LASSORegression:
    def __init__(self, lamb, X, Y):
        self.lamb = lamb
        self.X = X.T
        self.Y = array(Y)
        self.n, self.d = X.shape

    def _objective(self, betas):
        """The objective function"""
        return 1/self.n * norm(self.Y - (self.X.T @ betas))**2 + self.lamb * abs(betas).sum()

    def _partial_min_solution(self, betas, j):
        """Solution to the partial minimization function"""
        beta_j = betas[j]
        if beta_j == 0: return 0
        beta_without_j = delete(betas, j, axis=0)
        X_without_j = delete(self.X, j, axis=0)  # remove the jth column
        X_j = self.X[j]
        R_without_j = (self.Y - (X_without_j.T @ beta_without_j))
        Z_j = sum(X_j**2)

        indicator = 2/self.n * (X_j @ R_without_j)
        if abs(indicator) < self.lamb:
            return 0
        elif indicator <= -self.lamb:
            sign = -1
        else:  # indicator >= self.lamb
            sign = 1
        return ((sign * self.lamb) + indicator) / (2/self.n * Z_j)

    def cyclic_coord_descent(self, max_cycles=10, verbose=False, optimize=False):
        betas = normal(loc=0.00001, scale=0.0000001, size=self.d)  # init betas to small nonzero numbers
        beta_history = []
        objective_history = [] if optimize else [self._objective(betas)]
        for cycle in range(max_cycles):
            if verbose:
                print("Starting cycle {}".format(cycle+1))
            for j in range(self.d):
                betas[j] = self._partial_min_solution(betas, j)
            if not optimize:
                objective_history.append(self._objective(betas))
                beta_history.append(betas)
        if verbose:
            print("Cyclic coordinate descent complete")
        return betas, beta_history, objective_history

    def _random_coords(self):
        """Returns the indices of the coordinates in random order"""
        coords = array(range(0, self.d))
        shuffle(coords)
        return coords

    def random_coord_descent(self, max_cycles = 10, verbose=False, optimize=False):
        betas = normal(loc=0.00001, scale=0.0000001, size=self.d)  # init betas to small nonzero numbers
        objective_history = [] if optimize else [self._objective(betas)]
        beta_history = []
        for cycle in range(max_cycles):
            if verbose:
                print("Starting cycle {}".format(cycle+1))
            for j in self._random_coords():
                betas[j] = self._partial_min_solution(betas, j)
            if not optimize:
                objective_history.append(self._objective(betas))
                beta_history.append(betas)
        if verbose:
            print("Random coordinate descent complete")
        return betas, beta_history, objective_history

    @classmethod
    def predict(cls, betas, X):
        return X.T @ betas

