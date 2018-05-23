from .classifier import Classifier
from pandas import DataFrame
from numpy import unique, array
from multiprocessing import Pool

_classifier_combos = None
_classifier_generator = None
X, Y, epislon = None, None, 0
_train_args = None
_coefs = None
_coefs_ovr = None
_newX = None


def train_model(classes):
    global _X
    global _Y
    global _classifier_generator
    global _epsilon
    global _train_args
    print("Training linear classifier for classes: {}, {}".format(classes[0], classes[1]))
    these_rows = [y in (classes[0], classes[1]) for y in _Y]
    adjusted_Y = [1 if y == classes[0] else -1 for y in _Y[these_rows]]
    beta, beta_hist, _, _ = _classifier_generator(_X[these_rows], adjusted_Y).train(
        _epsilon,
        **_train_args
    )
    return beta


def train_one_vs_one(make_classifier_function, X, Y, epsilon, n_jobs=1, **kwargs):
    global _classifier_combos
    global _classifier_generator
    _classifier_generator = make_classifier_function
    global _X
    _X = X
    global _Y
    _Y = Y
    global _epsilon
    _epsilon = epsilon
    global _train_args
    _train_args = kwargs
    global _coefs
    classes = unique(Y)
    _classifier_combos = list()
    n_classes = len(classes)
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            _classifier_combos.append((classes[i], classes[j]))
    with Pool(n_jobs) as pool:
        print("Starting %d jobs in a multiprocessing pool" % n_jobs)
        all_betas = pool.map(train_model, _classifier_combos)
    _coefs =  dict(zip(_classifier_combos, all_betas))
    return _coefs


def train_one_vs_rest(make_classifier_function, X, Y, epsilon, n_jobs=1, **kwargs):
    global _coefs_ovr
    classes = unique(Y)
    trained_model = dict()
    for c in classes:
        y_tmp = array([1 if y == c else -1 for y in Y])
        print("Training OVR linear classifier for class: {}".format(c))
        beta, beta_hist, _, _ = make_classifier_function(X, y_tmp).train(
            epsilon,
            **kwargs
        )
        trained_model[c] = beta
    _coefs_ovr = trained_model
    return _coefs_ovr


def _predict_class(classes):
    global _newX
    global _coefs
    classifications = Classifier.classify(X, _coefs[(classes[0], classes[1])])
    return [classes[0] if t else classes[1] for t in classifications]


def predict(newX=None, n_jobs=1):
    global _X
    global _Y
    global _newX
    global _classifier_combos
    _newX = _X if newX is None else newX
    with Pool(n_jobs) as pool:
        predictions = pool.map(_predict_class, _classifier_combos)
    preds = DataFrame(predictions, columns=range(0, max(_Y)))
    return preds.mode(axis="columns")[0]


def predict_ovr(newX=None):
    global _X
    global _Y
    global _newX
    _newX = _X if newX is None else newX
    preds = DataFrame()
    classes = unique(_Y)
    idx = 0
    for c in classes:
        preds[idx] = Classifier.classify(X, _coefs_ovr[c])
        idx += 1
    return array([classes[i] for i in preds.idxmax(axis="columns")])
