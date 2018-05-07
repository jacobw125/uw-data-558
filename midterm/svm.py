from .hingeloss import HingeLossClassifier
from pandas import DataFrame
from numpy import unique


class HingeLossLinearSVM:
    def __init__(self, lambduh: float, X, Y):
        self.X = X
        self.Y = Y
        self.lambduh = lambduh
        self._coefs = None
        classes = unique(Y)
        self.classifier_combos = list()
        n_classes = len(classes)
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                self.classifier_combos.append((classes[i], classes[j]))

    def train(self, epsilon: float, **train_options):
        trained_model = dict()
        beta_history = dict()
        for class_i, class_j in self.classifier_combos:
            print("Training linear classifier for classes: {}, {}".format(class_i, class_j))
            these_rows = [y in (class_i, class_j) for y in self.Y]
            adjusted_Y = [1 if y == class_i else -1 for y in self.Y[these_rows]]
            beta, beta_hist, _, _ = HingeLossClassifier(self.lambduh, self.X[these_rows], adjusted_Y).train(
                epsilon,
                **train_options
            )
            trained_model[(class_i, class_j)] = beta
            beta_history[(class_i, class_j)] = beta_hist
        self._coefs = trained_model
        return trained_model, beta_history

    def predict(self, newX=None, custom_beta=None):
        X = self.X if newX is None else newX
        beta = self._coefs if custom_beta is None else custom_beta
        preds = DataFrame()
        idx = 0
        for class_i, class_j in self.classifier_combos:
            classifications = HingeLossClassifier.classify(X, beta[(class_i, class_j)])
            preds[idx] = [class_i if t else class_j for t in classifications]
            idx += 1
        return preds.mode(axis="columns")[0]
