from numpy import ndarray, array, exp, log, diag, identity, zeros
from numpy.linalg import norm, eigh


class LogisticRegression:
    def __init__(self, lamb, X, Y):
        self.lamb = lamb  # regularization parameter, assumed constant for the entire model fitting process
        self.X, self.Y = X.T, array(Y)  # note the transpose here to put it in terms matching class
        self.p, self.n = self.X.shape

    def _objective(self, beta):
        """Computes the objective function for a given set of betas"""
        return 1/self.n * sum(log(1 + exp(-self.Y * (self.X.T @ beta)))) + self.lamb*(beta.T @ beta)

    def _objective_long_way(self, beta):
        likelihood = 0
        for i in range(self.n):
            likelihood += log(1 + exp(-self.Y[i] * self.X[:,i] @ beta))
        return 1/self.n * likelihood + self.lamb*(norm(beta)**2)

    def _grad(self, beta):
        p_vector = 1/(1+exp(-self.Y * (self.X.T @ beta)))
        P = identity(self.n) - diag(p_vector)
        return (2*self.lamb*beta) - (1/self.n * self.X @ P @ self.Y)

    def _grad_long_way(self, beta):
        p_terms = list()
        for i in range(self.n):
           p_terms.append(
               1 / (1 + exp(
                   -self.Y[i] * (self.X[:,i].T @ beta)
               ))
           )
        P = identity(self.n) - diag(p_terms)
        return (2*self.lamb*beta) - (1/self.n * self.X @ P @ self.Y)

    def _backtrack(self, betas, grad_betas, init_t=1, alpha=0.5, beta=0.5, max_iter=1000):
        """Calculates the optimal stepsize via backtracking, where we first assess the objective at
        betas + grad_betas*init_t, then reduce t until we find a point where the objective function decreases
        by 'enough' to return t."""
        t = init_t
        norm_grad_betas = norm(grad_betas)
        for i in range(max_iter):
            if self._objective(betas - t*grad_betas) >= (self._objective(betas) - alpha*t*(norm_grad_betas**2)):
                t *= beta
            else:
                return t
        print('Max iterations of backtracking reached (%d)' % max_iter)
        return t

    def do_grad_descent(self, init_stepsize, epsilon, max_iter=1000):
        """Performs gradient descent and returns the final betas and two arrays, one history of betas and one history
        of the objective function over the optimization process."""
        print("Starting gradient descent with initial stepsize %f and epsilon %f" % (init_stepsize, epsilon))
        beta = zeros(self.p)
        beta_hist = [beta]
        objective_hist = [self._objective(beta)]

        i = 0
        t = None
        grad_beta = self._grad(beta)
        grad_beta_norm = norm(grad_beta)
        while grad_beta_norm > epsilon:
            if i % 50 == 0:
                print("%d: grad norm: %f > %f (%f)" % (i, grad_beta_norm, epsilon, self._objective(beta)))
            if i > max_iter:
                raise ValueError("Gradient descent failed to converge within %d iterations" % max_iter)
            t = self._backtrack(beta, grad_beta, init_t=init_stepsize)
            beta = beta - (t * grad_beta)
            grad_beta = self._grad(beta)
            grad_beta_norm = norm(grad_beta)
            beta_hist.append(beta)
            objective_hist.append(self._objective(beta))
            i += 1
        return beta, beta_hist, objective_hist

    def do_fastgrad_old(self, init_stepsize, epsilon, max_iter=100):  # still really slow for some reason
        """Performs fast gradient descent and returns the final betas and three arrays: one history of betas, one
        history of thetas, and one history of the objective function over the optimization process.."""
        print("Starting fast gradient descent with initial stepsize %f and epsilon %f" % (init_stepsize, epsilon))
        beta = zeros(self.p)
        theta = zeros(self.p)
        beta_hist = [beta]
        theta_hist = [theta]
        objective_hist = [self._objective(beta)]

        i = 0
        t = None
        grad_beta = self._grad(beta)
        while norm(grad_beta) > epsilon:
            if i % 50 == 0:
                print("%d: %f > %f (objective: %f)" % (i, norm(grad_beta), epsilon, self._objective(beta)))
            if i > max_iter:
                raise ValueError("Fast gradient failed to converge in %d iterations" % max_iter)
            t = self._backtrack(beta, grad_beta, init_t=init_stepsize)
            grad_theta = self._grad(theta)
            prev_beta = beta
            beta = theta - t*grad_theta
            beta_hist.append(beta)
            theta = i/(i+3)*(beta - prev_beta)
            theta_hist.append(theta)
            grad_beta = self._grad(beta)
            objective_hist.append(self._objective(beta))
            i += 1
        return beta, beta_hist, theta_hist, objective_hist

    def do_fastgrad(self, epsilon, init_stepsize=None, max_iter=100, optimize=True):
        """Performs fast gradient descent and returns the final betas and three arrays: one history of betas, one
        history of thetas, and one history of the objective function over the optimization process.."""

        if init_stepsize is None:
            init_stepsize = self.estimate_init_stepsize()

        print("Starting fast gradient descent with initial stepsize %f and epsilon %f" % (init_stepsize, epsilon))
        beta = zeros(self.p)
        theta = zeros(self.p)

        beta_hist = [beta]
        theta_hist = [theta]
        objective_hist = [self._objective(beta)] if not optimize else None

        i = 0
        t = init_stepsize
        grad_beta = self._grad(beta)
        while norm(grad_beta) > epsilon:
            if i % 50 == 0 and not optimize:
                print("%d: %f > %f (objective: %f)" % (i, norm(grad_beta), epsilon, self._objective(beta)))
            if i > max_iter:
                raise ValueError("Fast gradient failed to converge in %d iterations" % max_iter)
            t = self._backtrack(beta, grad_beta, init_t=t)
            beta_new = theta - t*self._grad(theta)
            theta = beta_new + i/(i+3)*(beta_new - beta)
            beta = beta_new
            grad_beta = self._grad(beta_new)

            if not optimize:
                beta_hist.append(beta)
                theta_hist.append(theta)
                objective_hist.append(self._objective(beta))
            i += 1
        return beta, beta_hist, theta_hist, objective_hist

    def estimate_init_stepsize(self):
        eigenvalues, eigenvectors = eigh(1/self.n * self.X @ self.X.T)
        lipschitz =  max(eigenvalues) + self.lamb
        return 1/lipschitz
