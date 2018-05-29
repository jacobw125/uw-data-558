import numpy as np
from numba import jit
from threading import Lock
from concurrent.futures import ThreadPoolExecutor


class BlockCDLasso:
    def __init__(self, lam, X, Y):
        """
        This class fits a Lasso penalized regression model using random block coordinate descent, allowing us to
        parallelize across the coordinates.
        :param lam: Lambda, tunes the serverity of the lasso penalty
        :param X: The independent variable matrix, assumed to be of shape (n x d)
        :param Y: Dependent variable
        """
        self.n, self.d = X.shape
        self.X = X.T
        self.Y = np.array(Y)
        self.block_locks = dict()  # a dict mapping block indexes to thread locks
        self.block_idxs = dict()  # a dict mapping block indexes to list of coordinate indices
                                  # (since coordinates are shuffled in each iteration)
        self.blocks = dict()  # a dict mapping block indexes to np.arrays for the betas in that block
        self.lam = lam
        self.optimize = True

    @property
    def betas(self):
        """Collects beta coefficients out of the blocks structure and returns them as an np.array. Thread safe."""
        b = np.zeros(self.d)
        if len(self.blocks) == 0: return np.array(list())
        for block in np.sort(np.array(list(self.blocks.keys()))):  # sorry for the mess - map keys are not always sorted
            with self.block_locks[block]:
                for i, j in enumerate(self.block_idxs[block]):
                    b[j] = np.copy(self.blocks[block][i])
        return np.array(b)

    @jit
    def _objective(self):
        """The objective function."""
        likelihood = np.linalg.norm(self.Y - (self.X.T @ self.betas))**2  # standard sum square residuals
        penalty = np.abs(self.betas).sum()  # L1-norm penalty
        return 1/self.n * likelihood + self.lam * penalty

    @jit
    def _partial_min_solution(self, j):
        """This returns the solution to the partial minimization function with respect to the jth coordinate"""
        beta_without_j = np.delete(self.betas, j, axis=0)
        X_without_j = np.delete(self.X, j, axis=0)
        X_j = self.X[j]  # these are the X values for the jth feature in the model
        # Make predictions and obtain residuals on the full set of Ys, without the effect of the jth predictor included
        R_without_j = (self.Y - (beta_without_j.T @ X_without_j))
        c_j = 2/self.n * (X_j @ R_without_j)  # This quantity is described in the notes
        # The following if statements are due to the subgradient of the L1 penality
        if abs(c_j) <= self.lam:  # this step is what causes the lasso to shrink coefficients to 0 based on lambda
            return 0
        a_j = 2 * sum(X_j**2)  # also described in notes
        if c_j < -self.lam:
            return (c_j + self.lam) / (a_j / self.n)
        elif c_j > self.lam:
            return (c_j - self.lam) / (a_j / self.n)

    def _update_block(self, block_idx):
        """
        This method performs coordinate updates on a block of coordinates at a time. Thread safe.
        :param block_idx: The index of the block to perform sequential updates on. Must be a key in in self.blocks.
        :return: None, updates the self.betas array (thread-safe).
        """
        my_coord_indices = self.block_idxs[block_idx]
        for i,j in enumerate(my_coord_indices):
            soln = self._partial_min_solution(j)
            with self.block_locks[block_idx]:
                self.blocks[block_idx][i] = soln

    def fit(self, max_cycles=10, n_blocks=8, pool_size=8, optimize=False):
        """Perform the model fitting procedure (block random coordinate descent) described in the README.
        :param max_cycles: The number of complete iterations to perform. A complete iteration is an update to every
                           coordinate.
        :param n_blocks: The number of blocks to split the coordinate set into.
        :param pool_size: The number of workers to create in the multiprocessing Pool that will process each iteration.
        :param optimize: If true, print diagnostic information while fitting, and returns full beta_history and
                         objective_history objects with the fitted coefficients.
        """
        self.optimize = optimize

        # Init betas to small random values, set up blocks, block indexes, and locks. This loop looks complex, but
        # it's really just making sure that self.d % n_blocks != 0, we take care of the extra coordinates.
        i, j = 0, 0
        block_size = int(np.floor(self.d / n_blocks))
        if block_size == 0:
            block_size = 1
        coeffs_remaining = self.d
        while coeffs_remaining > 0:
            n_elements_in_this_block = block_size if coeffs_remaining >= block_size else coeffs_remaining
            self.blocks[j] = np.random.normal(
                loc=0.00001,
                scale=0.0000001,
                size=n_elements_in_this_block
            )
            self.block_idxs[j] = range(i, i+n_elements_in_this_block)
            self.block_locks[j] = Lock()
            i += block_size
            j += 1
            coeffs_remaining -= block_size

        objective_history = [] if optimize else [self._objective()]
        beta_history = []
        for cycle in range(max_cycles):
            # Shuffle the coefficients into different blocks and rearrange the data structures
            all_betas = self.betas
            idxs = list(range(len(all_betas)))
            np.random.shuffle(idxs)
            for j in range(int(np.ceil(i/block_size))):
                start = j*block_size
                end = start + min(len(idxs) - start, block_size)
                this_block_beta_indexes = idxs[start:end]
                self.blocks[j] = all_betas[this_block_beta_indexes]
                self.block_idxs[j] = this_block_beta_indexes

            # ThreadPoolExecutor allows me to use locks but at the cost of not working in a true forked process
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                for i in range(n_blocks):
                    executor.submit(self._update_block, i)

            if not optimize:
                objective_history.append(self._objective())
                beta_history.append(np.copy(self.betas))
        return self.betas, beta_history, objective_history

    @staticmethod
    def predict(X, betas):
        return X @ betas
