from .classifier import Classifier
from pandas import DataFrame
from numpy import unique, array, sort as np_sort, random, ravel, vstack
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


def train_ovr_model(target_class):
    global _X
    global _Y
    global _classifier_generator
    global _epsilon
    global _train_args
    print("Training OVR classifier for class {}".format(target_class))
    is_class = array([y == target_class for y in _Y])
    x_out_of_class = _X[~is_class]
    y_out_of_class = _Y[~is_class]
    random_out_of_class_subset = random.choice(range(len(x_out_of_class)), size=is_class.sum())
    x_out_of_class = x_out_of_class[random_out_of_class_subset]
    y_out_of_class = y_out_of_class[random_out_of_class_subset]
    x_in_class = _X[is_class]
    y_in_class = _Y[is_class]
    new_X = vstack((x_out_of_class, x_in_class))
    new_y = [1 if y == target_class else -1 for y in ravel(vstack((y_out_of_class, y_in_class)))]
    beta = _classifier_generator(new_X, new_y).train(_epsilon, **_train_args)
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
    classes = np_sort(unique(Y))
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
    global _coefs_ovr
    classes = np_sort(unique(Y))
    with Pool(n_jobs) as pool:
        betas = pool.map(train_ovr_model, classes)
    _coefs_ovr = dict(zip(classes, betas))
    return _coefs_ovr


def _predict_class(classes):
    global _newX
    global _coefs
    print("Predicting %s vs %s" % classes)
    classifications = Classifier.classify(_newX, _coefs[(classes[0], classes[1])])
    return array([classes[0] if t else classes[1] for t in classifications])


def predict(newX=None, n_jobs=1):
    global _X
    global _Y
    global _newX
    global _classifier_combos
    _newX = _X if newX is None else newX
    with Pool(n_jobs) as pool:
        predictions = pool.map(_predict_class, _classifier_combos)
    preds = DataFrame(dict(zip(["%s vs %s" % (c1, c2) for c1, c2 in _classifier_combos], predictions)))
    return preds.mode(axis="columns")[0]


def _predict_class_ovr(target_class):
    global _newX
    global _coefs_ovr
    print("Predicting OVR for %s" % target_class)
    return _newX @ _coefs_ovr[target_class]


def predict_ovr(newX=None, n_jobs=1):
    global _X
    global _Y
    global _newX
    global _coefs_ovr
    _newX = _X if newX is None else newX
    classes = np_sort(unique(_Y))
    with Pool(n_jobs) as pool:
        preds = pool.map(_predict_class_ovr, classes)
    preds = DataFrame(dict(zip(classes, preds)))
    return array([int(x) for x in preds.idmax(axis="columns")[0]])
