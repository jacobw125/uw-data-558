from typing import Callable
import numpy as np
from numba import jit
from multiprocessing import Pool
from pandas import DataFrame


def computegram(X, kernel: Callable):
    n, p = X.shape
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel(X[i], X[j])
    return K

def poly_kernel(x, y, p=7, d=1):
    return ((x.T @ y) + d)**p

def linear_kernel(x,y):
    return x.T @ y

def gaussian_rbf_kernel(x, y, sigma=1):
    return np.exp(-(np.linalg.norm(x-y)**2) / (2*sigma**2))

def kerneleval(X, newX, kernel):
    '''Assumes X is (n x d) and newX is a single new datapoint'''
    return np.apply_along_axis(lambda xi: kernel(xi, newX), axis=1, arr=X)

def train_huberized_hinge_kernel_classifier(X, y, kernel, lambduh, init_stepsize, epsilon, max_iter=2000, bt_max_iter=100, optimize=False):
    n = len(y)
    K=computegram(X, kernel)

    @jit
    def obj(alphas):
        l_hh = 0
        Kalpha = K @ alphas
        for i in range(n):
            yt = y[i] * Kalpha[i]
            if yt > 1.5: continue
            elif yt <= 0.5:
                l_hh += 1-yt
            else:   # abs(1-yt) <= 0.5
                l_hh += (1.5-yt)**2 / 2
        penalty = alphas.T @ Kalpha
        return 1/n * l_hh + lambduh*penalty

    @jit
    def grad(alphas):
        grad_l_hh = 0
        Kalpha = K @ alphas
        for i in range(n):
            yt = y[i] * Kalpha[i]
            if yt > 1.5: continue
            elif yt <= 0.5:
                grad_l_hh += K[i,:] * (-y[i])
            else:  # abs(1-yt) <= 0.5
                grad_l_hh += (-y[i]*K[i,:])*(1.5 - yt)
        grad_penalty = 2*lambduh*Kalpha
        return 1/n*grad_l_hh + grad_penalty

    def backtrack(alphas, grad_alphas, init_t=1, alpha=0.5, beta=0.5):
        t = init_t
        norm_grad_alphas = np.linalg.norm(grad_alphas)
        for i in range(bt_max_iter):
            if obj(alphas - t*grad_alphas) >= (obj(alphas) - alpha*t*(norm_grad_alphas**2)):
                t *= beta
            else:
                return t
        print('Max iterations of backtracking reached (%d)' % (bt_max_iter))
        return -1
        return t

    ## Now use these functions to perform fast gradient descent

    if not optimize:
        print("Starting fast gradient descent on %d observations with lambda %f, initial stepsize %f, and epsilon %f" % (
            n, lambduh, init_stepsize, epsilon
        ))
    alpha = np.zeros(n)
    theta = np.zeros(n)

    if not optimize:
        alpha_hist = [alpha]
        theta_hist = [theta]
        objective_hist = [obj(alpha)]

    i = 0
    t = init_stepsize
    grad_alpha = grad(alpha)
    while np.linalg.norm(grad_alpha) > epsilon:
        if i % 500 == 0 and not optimize:
            print("%d: %f > %f (objective: %f)" % (i, np.linalg.norm(grad_alpha), epsilon, obj(alpha)))
        if i > max_iter:
            break
        t = backtrack(alpha, grad_alpha, init_t=t)
        if t == -1:
            break
        alpha_new = theta - t*grad(theta)
        theta = alpha_new + i/(i+3)*(alpha_new - alpha)
        alpha = alpha_new
        grad_alpha = grad(alpha_new)

        if not optimize:
            alpha_hist.append(np.copy(alpha))
            theta_hist.append(np.copy(theta))
            objective_hist.append(obj(alpha))
        i += 1
    if not optimize:
        print("Converged in {} iterations".format(i))
        return alpha, alpha_hist, theta_hist, objective_hist
    return alpha

def train_one_vs_one(classes):
    global _X
    global _y
    global _lambduh
    global _kernel
    global _init_stepsize
    global _max_iter
    global _max_iter_bt
    print("Training classifier for classes: {}, {}".format(classes[0], classes[1]))
    these_rows = [y in (classes[0], classes[1]) for y in _y]
    adjusted_Y = [1 if y == classes[0] else -1 for y in _y[these_rows]]
    alphas = train_huberized_hinge_kernel_classifier(
        _X[these_rows], adjusted_Y, _kernel, _lambduh, _init_stepsize, _epsilon, _max_iter, _max_iter_bt, optimize=True
    )
    return alphas

def mysvm(X, y, kernel, lambduh, init_stepsize, epsilon, max_iter=1000, max_iter_bt=500, n_jobs=7):
    global _X  # this is to get around the pickle-able requirement in the multiprocessing.Pool.map function
    _X = X
    global _y
    _y = y
    global _lambduh
    _lambduh = lambduh
    global _epsilon
    _epsilon=epsilon
    global _kernel
    _kernel = kernel
    global _init_stepsize
    _init_stepsize = init_stepsize
    global _max_iter
    _max_iter = max_iter
    global _max_iter_bt
    _max_iter_bt = max_iter_bt

    classes = np.sort(np.unique(y))
    classifier_combos = list()
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            classifier_combos.append((classes[i], classes[j]))
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            print("Starting %d training jobs" % n_jobs)
            all_alphas = pool.map(train_one_vs_one, classifier_combos)
    else:
        all_alphas = [train_one_vs_one(c) for c in classifier_combos]
    coefs =  dict(zip(classifier_combos, all_alphas))
    return coefs

def predict_classifier(classes):
    global _train_data
    global _train_labels
    global _newdata
    global _kernel
    global _alphas
    global digits
    global digit_labels
    train = _train_data[[x in classes for x in _train_labels]]
    print("Predicting %s vs %s" % classes)
    return [
        classes[0] if
        (kerneleval(train, _newdata[i] , _kernel)  @ _alphas[classes]) > 0
        else classes[1]
        for i in range(len(_newdata))
    ]

def predict_svm(train_data, train_labels, newdata, alphas, kernel, n_jobs=1):
    global _train_data
    _train_data = train_data
    global _train_labels
    _train_labels = train_labels
    global _newdata
    _newdata = newdata
    global _alphas
    _alphas = alphas
    global _kernel
    _kernel = kernel
    classes = np.sort(np.unique(train_labels))
    classifier_combos = list()
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            classifier_combos.append((classes[i], classes[j]))
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            predictions = pool.map(predict_classifier, classifier_combos)
    else:
        predictions = [predict_classifier(c) for c in classifier_combos]
    preds = DataFrame(dict(zip(["%s vs %s" % (c1, c2) for c1, c2 in classifier_combos], predictions)))
    return np.array([int(x) for x in preds.mode(axis="columns")[0]])

