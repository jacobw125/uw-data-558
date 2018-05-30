# Polished Code Release
Jacob Warwick, for DATA 558 A Sp 18: Statistical Machine Learning For Data Scientists

# Overview
This code is an implementation of the random block coordinate descent algorithm for fitting a penalized lasso regression model. 

Under traditional coordinate descent, we use a partial minimization solution to perform updates to a single coordinate at a time, rotating through the coordinates in either a cyclic or a stochastic pattern until all coordinates have been updated (constituting one iteration). Since each update relies on the coordinate updates that were already performed in that iteration, the algorithm is entirely sequential and cannot be parallelized.

Lasso regression is often used for situations where the number of parameters are very high. This means that there is a strong motivation to be able to scale up the number of coordinates we can fit in such a model without having to linearly scale the fitting time. Leveraging parallelization allows us to speed up model fitting in situations where there is high dimensionality, by throwing more processors at the problem. 

Under random block coordinate descent, we shuffle the coordinates and then split that set into a number of groups (blocks). The blocks are trained simultaneously, broadcasting their updates after each coordinate is updated.

My intuition is that as the number of blocks increases, this algorithm requires more iterations to converge compared to the regular coordinate descent, because at the time of each coordinate update, fewer of the prior coordinates will have been updated.

This approach is discussed in much more detail in this paper:

[Xu, Yangyang and Yin, Wotao: A Block Coordinate Descent Method for Regularized Multiconvex Optimization
 with Applications to Nonnegative Tensor Factorization and Completion](http://www.math.ucla.edu/~wotaoyin/papers/pdf/Xu%20and%20Yin%20-%202013%20-%20A%20Block%20Coordinate%20Descent%20Method%20for%20Regularized.pdf)

There are probably some assumptions about the convexity of the problem space with respect to each block that I have not examined.

# Code and examples
* **block_cd_Lasso.py**: My implementation.
* **block_cd_lasso_threads_only.py**: An attempt I made that uses threads only instead of processes. Currently returning slightly incorrect results due to a subtle concurrency bug I didn't have time track down.
* **demo.py**: Simulates data, trains a model on that data, then compares the coefficients to Scikit learn's Lasso method. Then fits a model on the Spam dataset and predicts / assesses the holdout set classification accuracy.

Data are packaged with the repository.