from numpy import load, array, save
from scipy.special import expit
from hw3 import RidgeRegression

train = load("train_features.npy")
train_labels = load("train_labels.npy")

target_cat, other_cat = 2, 5
subset = (train_labels == target_cat) | (train_labels == other_cat)
X = train[subset]
Y = array([1 if x==target_cat else -1 for x in train_labels[subset]])
r = RidgeRegression(1, X, Y)
betas, _, _ = r.do_grad_descent(r.estimate_init_stepsize(), 0.01)


predictions = expit(X @ betas)
prediction_bools = array([1 if x > 0.5 else 0 for x in predictions])
save("betas.npy", betas)
save("prediction_bools.npy", prediction_bools)
