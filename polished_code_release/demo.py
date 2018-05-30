from block_cd_lasso_threaded import BlockCDLasso
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from pandas import read_csv

print("Demo: Regression on simulated data")
cov_matrix = np.identity(200)   # Simulate 200 columns of data with correlations
cov_matrix += 0.8
for i in range(200): cov_matrix[i,i] = 1
sim_data = np.vstack(np.random.multivariate_normal(
    mean=np.array(range(0, 200)),
    cov=cov_matrix,
    size=(1000, 200)
))
X = StandardScaler().fit_transform(sim_data[:, 0:199])
y = sim_data[:, 199]
model = BlockCDLasso(0.01, X, y)
print("Starting coordinate descent")
betas, beta_hist, objective_hist = model.fit(max_cycles=100, n_blocks=10, pool_size=10, optimize=False)
print("Objective descent history: %s\n" % objective_hist)

skmodel = Lasso(alpha=0.001 * 1000, fit_intercept=False)
skmodel.fit(X, y)
print("Difference in coefficients between this approach and scikit: %s\nMean difference: %f\n" %
      (betas - skmodel.coef_, np.mean(betas - skmodel.coef_)))

print("\n\n==================\n\nDemo: classification on the Spam dataset")
spam = read_csv("data/spam.csv")
last_col = spam.columns[-1]
Y = (spam[last_col] * 2)-1  # convert to 1/-1
spam = spam.drop(last_col, axis=1)
stdscaler = StandardScaler().fit(spam)
X = stdscaler.transform(spam)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = BlockCDLasso(0.001, X_train, y_train)
betas, beta_hist, objective_hist = model.fit(max_cycles=20, n_blocks=4, pool_size=4, optimize=False)
print("Objective history: %s" % objective_hist)
predictions = np.array([1 if x > 0 else -1 for x in BlockCDLasso.predict(X_test, betas)])
print("Classification accuracy on holdout set: %f" % (predictions == y_test).mean())
