import numpy as np
from numba import jit
from multiprocessing import Queue, Process, Lock
from time import sleep


def block_update_worker(lam, X, Y, blocks, block_idxs, block_locks, queue: Queue):
    """This function is the parallelizable worker that updates a set of coefficients."""
    d, n = X.shape

    def _get_betas():
        """Recomposes the betas metrix from the block data structure"""
        betas = np.zeros(d)
        for block_idx in np.sort(list(block_idxs.keys())):
            with block_locks[block_idx]:
                for i, j in enumerate(block_idxs[block_idx]):
                    betas[j] = np.copy(blocks[block_idx][i])
        return betas

    @jit
    def _partial_min_solution(j):
        """This returns the solution to the partial minimization function with respect to the jth coordinate"""
        betas = _get_betas()
        beta_without_j = np.delete(betas, j, axis=0)
        X_without_j = np.delete(X, j, axis=0)
        X_j = X[j]  # these are the X values for the jth feature in the model
        # Make predictions and obtain residuals on the full set of Ys, without the effect of the jth predictor included
        R_without_j = (Y - (beta_without_j.T @ X_without_j))
        c_j = 2/n * (X_j @ R_without_j)  # This quantity is described in the notes
        # The following if statements are due to the subgradient of the L1 penality
        if abs(c_j) <= lam:  # this step is what causes the lasso to shrink coefficients to 0 based on lambda
            return 0
        a_j = 2 * sum(X_j**2)  # also described in notes
        if c_j < -lam:
            return (c_j + lam) / (a_j / n)
        elif c_j > lam:
            return (c_j - lam) / (a_j / n)

    def update_block(target_idx):
        my_coord_indices = block_idxs[target_idx]
        for i,j in enumerate(my_coord_indices):
            soln = _partial_min_solution(j)
            with block_locks[target_idx]:
                blocks[target_idx][i] = soln

    while True: # act a a worker in a pool
        next_block_to_update = queue.get()
        if next_block_to_update is None: return
        update_block(next_block_to_update)


class BlockCDLasso:
    def __init__(self, lam, X, Y):
        """
        This class fits a Lasso penalized regression model using random block coordinate descent, allowing us to
        parallelize across the coordinates.
        :param lam: Lambda, tunes the severity of the lasso penalty
        :param X: The independent variable matrix, assumed to be of shape (n x d)
        :param Y: Dependent variable
        """
        self.n, self.d = X.shape
        self.X = X.T
        self.Y = np.array(Y)
        self.block_idxs = dict()  # a dict mapping block indexes to list of coordinate indices using Array
                                  # (since coordinates are shuffled in each iteration)
        self.block_locks = dict()  # restricts read/write operations on each block
        self.blocks = dict()  # a dict mapping block indexes to arrays for the betas in that block
        self.lam = lam
        self.optimize = True

    @property
    def betas(self):
        """Collects beta coefficients out of the blocks structure and returns them as an np.array."""
        b = np.zeros(self.d)
        if self.blocks is None or len(self.blocks) == 0: return np.array(list())
        for block in np.sort(np.array(list(self.blocks.keys()))):  # Map keys are not always sorted.
            #  Blocks not expected to change over the course of training, so the above line is not a concurrency bug
            with self.block_locks[block]:
                for i, j in enumerate(self.block_idxs[block]):
                    b[j] = np.copy(self.blocks[block][i])
        return b

    @jit
    def _objective(self):
        """The objective function."""
        likelihood = np.linalg.norm(self.Y - (self.X.T @ self.betas))**2  # standard sum square residuals
        penalty = np.abs(self.betas).sum()  # L1-norm penalty
        return 1/self.n * likelihood + self.lam * penalty

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

        # Init the process pool
        queue = Queue()
        pool = [Process(
                    target=block_update_worker,
                    args=(self.lam, self.X, self.Y, self.blocks, self.block_idxs, self.block_locks, queue)
                ) for ignore in range(pool_size)]
        for worker in pool: worker.start()

        try:  # Main loop for model-fitting iterations
            for cycle in range(max_cycles):
                # Shuffle the coefficients into different blocks and rearrange the data structures
                all_betas = self.betas
                idxs = list(range(len(all_betas)))
                np.random.shuffle(idxs)
                n_actual_blocks = int(np.ceil(i/block_size))
                for j in range(n_actual_blocks ):
                    start = j*block_size
                    end = start + min(len(idxs) - start, block_size)
                    this_block_beta_indexes = idxs[start:end]
                    self.blocks[j] = all_betas[this_block_beta_indexes]
                    self.block_idxs[j] = this_block_beta_indexes

                for i in range(n_actual_blocks):
                    queue.put(i)
                while not queue.empty():  # this version of queue doesn't have a join() method
                    sleep(0.1) # so I use sleep to free up processor resources while we wait

                if not optimize:
                    objective_history.append(self._objective())
                    beta_history.append(np.copy(self.betas))
            return self.betas, beta_history, objective_history
        finally:  # Shut down the workers
            for ignore in pool: queue.put(None)
            for process in pool: process.join()  # waits for all processes to shut down

    @staticmethod
    def predict(X, betas):
        return X @ betas
