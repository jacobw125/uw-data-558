from numpy import array, zeros
from numpy.linalg import norm, eigh
from numba import jit
from .classifier import Classifier


class SmoothedHingeLossClassifier(Classifier):
    @jit
    def _obj(self, beta):
        likelihood = 0
        for i in range(self.n):
            yt = self.Y[i] * (self.X[:, i].T @ beta)
            if yt > 1.5:
                likelihood += 0
            elif abs(1-yt) <= 0.5:
                likelihood += ((1.5 - yt)**2)/2
            else:
                likelihood += 1 - yt

        return 1/self.n * likelihood + self.lambduh * (norm(beta)**2)

    @jit
    def _grad(self, beta):
        grad_likelihood = 0
        for i in range(self.n):
            yt = self.Y[i] * (self.X[:, i].T @ beta)
            if yt > 1.5:
                grad_likelihood += zeros(self.d)
            elif abs(1-yt) <= 0.5:
                grad_likelihood += -self.Y[i]*self.X[:, i] * (1.5 - yt)
            else:
                grad_likelihood += -self.Y[i] * self.X[:, i]

        return 1/self.n*grad_likelihood + 2*self.lambduh*beta
