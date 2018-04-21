from numpy import ndarray, array, exp, log, diag, identity
from numpy.linalg import norm, eigh


class RidgeRegression:
    def __init__(self, lamb, X, Y):
        self.lamb = lamb  # regularization parameter, assumed constant for the entire model fitting process
        self.X, self.Y = X.T, array(Y)  # note the transpose here to put it in terms matching class
        self.p, self.n = self.X.shape

    def _objective(self, beta):
        """Computes the objective function for a given set of betas"""
        likelihood = 0
        for i in range(self.n):
            likelihood += log(1 + exp(-self.Y[i] * self.X[:,i] @ beta))
        return 1/self.n * likelihood + self.lamb*(norm(beta)**2)

    def _grad(self, beta):
        """Computes the objective function gradient for a given set of betas"""
        p_terms = list()
        for i in range(self.n):
           p_terms.append(
               1 / (1 + exp(
                   -self.Y[i] * (self.X[:,i].T @ beta)
               ))
           )
        P = identity(self.n) - diag(p_terms)
        return (2*self.lamb*beta) - (1/self.n * self.X @ P @ self.Y)

    def _grad_long_way(self, beta):
        """Used to confirm _grad is working correctly"""
        diff_sum = array([0.0] * self.p)
        for i in range(self.n):
            x, y = self.X[:, i], self.Y[i]
            exp_term = exp(-y * x.T @ beta)
            diff_sum = diff_sum + ( y*x * (exp_term  / (1 + exp_term)) )
        return 2*self.lamb*beta - 1/self.n * diff_sum

    def _backtrack(self, betas, init_t=5, alpha=0.5, beta=0.5, max_iter=1000):
        """Calculates the optimal stepsize via backtracking, where we first assess the objective at
        betas + grad_betas*init_t, then reduce t until we find a point where the objective function decreases
        by 'enough' to return t."""
        t = init_t
        grad_betas = self._grad(betas)
        norm_grad_betas = norm(grad_betas)
        for i in range(max_iter):
            a = self._objective(betas - t*grad_betas)
            b = (self._objective(betas) - alpha*t*(norm_grad_betas**2))
            if a >= b:
                t *= beta
            else:
                return t
        print('Max iterations of backtracking reached (%d)' % max_iter)
        return t

    def do_grad_descent(self, init_stepsize, epsilon, max_iter=1000):
        """Performs gradient descent and returns the final betas and two arrays, one history of betas and one history
        of the objective function over the optimization process."""
        beta = array([0.0] * self.p)
        beta_hist = [beta]
        objective_hist = [self._objective(beta)]

        i = 0
        t = None
        grad_beta = self._grad(beta)
        grad_beta_norm = norm(grad_beta)
        while grad_beta_norm > epsilon:
            if i % 100 == 0:
                print("%d: grad norm: %f > %f (%f)" % (i, grad_beta_norm, epsilon, self._objective(beta)))
            if i > max_iter:
                raise ValueError("Gradient descent failed to converge within %d iterations" % max_iter)
            t = self._backtrack(beta, t or init_stepsize)
            beta = beta - (t * grad_beta)
            grad_beta = self._grad(beta)
            grad_beta_norm = norm(grad_beta)
            beta_hist.append(beta)
            objective_hist.append(self._objective(beta))
            i += 1
        return beta, beta_hist, objective_hist

    def do_fastgrad(self, init_stepsize, epsilon, max_iter=1000):
        """Performs fast gradient descent and returns the final betas and three arrays: one history of betas, one
        history of thetas, and one history of the objective function over the optimization process.."""
        beta = array([0.0] * self.p)
        theta = beta
        beta_hist = [beta]
        theta_hist = [theta]
        objective_hist = [self._objective(beta)]

        i = 0
        grad_beta = self._grad(beta)
        while norm(grad_beta) > epsilon:
            if i % 100 == 0:
                print("%d: %f > %f (objective: %f)" % (i, norm(grad_beta), epsilon, self._objective(beta)))
            if i > max_iter:
                raise ValueError("Fast gradient failed to converge in %d iterations" % max_iter)
            t = self._backtrack(beta, init_stepsize)
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

    def estimate_init_stepsize(self):
        """Returns the inverse of the estimated lipschitz constant."""
        eigenvalues, eigenvectors = eigh(1/self.n * self.X @ self.X.T)
        lipschitz =  max(eigenvalues) + self.lamb
        return 1/lipschitz
