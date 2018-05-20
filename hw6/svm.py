from .classifier import Classifier
from pandas import DataFrame
from numpy import unique, array


class SVM:
    def __init__(self, make_classifier_function, X, Y):
        self.X = X
        self.Y = Y
        self.__ClassifierGenerator = make_classifier_function
        self._coefs = None
        classes = unique(Y)
        self.classifier_combos = list()
        n_classes = len(classes)
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                self.classifier_combos.append((classes[i], classes[j]))

    def train(self, epsilon, **kwargs):
        trained_model = dict()
        for classes in self.classifier_combos:
            print("Training linear classifier for classes: {}, {}".format(classes[0], classes[1]))
            these_rows = [y in (classes[0], classes[1]) for y in self.Y]
            adjusted_Y = [1 if y == classes[0] else -1 for y in self.Y[these_rows]]
            beta, beta_hist, _, _ = self.__ClassifierGenerator(self.X[these_rows], adjusted_Y).train(
                epsilon,
                **kwargs
            )
            trained_model[classes] = beta
        self._coefs = trained_model
        return trained_model

    def train_one_vs_rest(self, epsilon, **kwargs):
        classes = unique(self.Y)
        trained_model = dict()
        for c in classes:
            y_tmp = array([1 if y == c else -1 for y in self.Y])
            print("Training OVR linear classifier for class: {}".format(c))
            beta, beta_hist, _, _ = self.__ClassifierGenerator(self.X, y_tmp).train(
                epsilon,
                **kwargs
            )
            trained_model[c] = beta
        self._coefs_ovr = trained_model
        return trained_model

    def predict(self, newX=None, custom_beta=None):
        X = self.X if newX is None else newX
        beta = self._coefs if custom_beta is None else custom_beta
        preds = DataFrame()
        idx = 0
        for class_i, class_j in self.classifier_combos:
            classifications = Classifier.classify(X, beta[(class_i, class_j)])
            preds[idx] = [class_i if t else class_j for t in classifications]
            idx += 1
        return preds.mode(axis="columns")[0]

    def predict_ovr(self, newX=None):
        X = self.X if newX is None else newX
        preds = DataFrame()
        classes = unique(self.Y)
        idx = 0
        for c in classes:
            preds[idx] = Classifier.classify(X, self._coefs_ovr[c])
            idx += 1
        return array([classes[i] for i in preds.idxmax(axis="columns")])
