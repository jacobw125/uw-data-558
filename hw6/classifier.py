from abc import ABC, abstractmethod
from numba import jit
from numpy import array, zeros
from numpy.linalg import norm, eigh


class Classifier(ABC):

    def __init__(self, lambduh: float, X, Y):
        self.n, self.d = X.shape
        self.X = X.T
        self.Y = array(Y)
        self.lambduh = lambduh

    @abstractmethod
    def _obj(self, beta): raise NotImplementedError()

    @abstractmethod
    def _grad(self, beta): raise NotImplementedError()

    def _backtrack(self, betas, grad_betas, init_t=1, alpha=0.5, beta=0.5, max_iter=1000):
        t = init_t
        norm_grad_betas = norm(grad_betas)
        for i in range(max_iter):
            if self._obj(betas - t*grad_betas) >= (self._obj(betas) - alpha*t*(norm_grad_betas**2)):
                t *= beta
            else:
                return t
        print('Max iterations of backtracking reached (%d) (%s)' % (max_iter, grad_betas))
        return t

    @jit
    def _estimate_init_stepsize(self):
        eigenvalues, eigenvectors = eigh(1/self.n * self.X @ self.X.T)
        lipschitz =  max(eigenvalues) + self.lambduh
        return 1/lipschitz

    def train(self, epsilon, init_stepsize=None, max_iter=100, max_bktrack_iter=100, optimize=True):
        if init_stepsize is None:
            init_stepsize = self._estimate_init_stepsize()

        if not optimize:
            print("Starting fast gradient descent with initial stepsize %f and epsilon %f" % (init_stepsize, epsilon))
        beta = zeros(self.d)
        theta = zeros(self.d)

        beta_hist = [beta]
        theta_hist = [theta]
        objective_hist = [self._obj(beta)] if not optimize else None

        i = 0
        t = init_stepsize
        grad_beta = self._grad(beta)
        while norm(grad_beta) > epsilon:
            if i % 50 == 0 and not optimize:
                print("%d: %f > %f (objective: %f)" % (i, norm(grad_beta), epsilon, self._obj(beta)))
            if i > max_iter:
                raise ValueError("Fast gradient failed to converge in %d iterations" % max_iter)
            t = self._backtrack(beta, grad_beta, init_t=t, max_iter=max_bktrack_iter)
            beta_new = theta - t*self._grad(theta)
            theta = beta_new + i/(i+3)*(beta_new - beta)
            beta = beta_new
            grad_beta = self._grad(beta_new)

            if not optimize:
                beta_hist.append(beta)
                theta_hist.append(theta)
                objective_hist.append(self._obj(beta))
            i += 1
        if not optimize:
            print("Converged in {} iterations".format(i))
        return beta, beta_hist, theta_hist, objective_hist

    @classmethod
    def classify(cls, newX, beta, cutpoint=0):
        return newX @ beta > cutpoint
