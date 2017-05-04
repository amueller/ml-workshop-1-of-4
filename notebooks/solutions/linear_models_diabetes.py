import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

# create dataframe for easy boxplot
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df.boxplot()

plt.figure()
plt.title("Target distribution")
plt.hist(diabetes.target, bins="auto")

X_train, X_test, y_train, y_test = train_test_split(diabetes.data,
                                                    diabetes.target)

scores_lr = cross_val_score(LinearRegression(), X_train, y_train, cv=10)
print("Linear regression score: {}".format(scores_lr.mean()))
scores_ridge = cross_val_score(Ridge(), X_train, y_train, cv=10)
print("Ridge Regression score: {}".format(scores_ridge.mean()))

# With scaled data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scores_lr = cross_val_score(LinearRegression(), X_train_scaled, y_train, cv=10)
print("Linear regression w/ scaling: {}".format(scores_lr.mean()))
scores_ridge = cross_val_score(Ridge(), X_train_scaled, y_train, cv=10)
print("Ridge regression w/ scaling: {}".format(scores_ridge.mean()))

from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': np.logspace(-3, 3, 7)}
grid = GridSearchCV(Ridge(), param_grid, cv=10)
grid.fit(X_train_scaled, y_train)

res = pd.DataFrame(grid.cv_results_)
res.plot("param_alpha", ["mean_train_score", "mean_test_score"], logx=True)
plt.title("Ridge grid search")


print(grid.best_params_, grid.best_score_)

lr = LinearRegression().fit(X_train_scaled, y_train)

plt.figure()
plt.title("Coefficients LR vs Ridge")
plt.hlines(0, 0, X_train.shape[1], linewidth=.5)
plt.plot(grid.best_estimator_.coef_, 'o', label="Ridge({})".format(grid.best_params_['alpha']))
plt.plot(lr.coef_, 'o', label="LR", alpha=.6)
plt.legend()

from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': np.logspace(-3, 3, 7)}
grid_lasso = GridSearchCV(Lasso(), param_grid, cv=10)
grid_lasso.fit(X_train_scaled, y_train)

res = pd.DataFrame(grid_lasso.cv_results_)
res.plot("param_alpha", ["mean_train_score", "mean_test_score"], logx=True)
plt.title("Lasso grid search")
print(grid_lasso.best_params_, grid_lasso.best_score_)

plt.figure()
plt.title("coefficients")
plt.hlines(0, 0, X_train.shape[1], linewidth=.5)
plt.plot(grid.best_estimator_.coef_, 'o', label="Ridge({})".format(grid.best_params_['alpha']))
plt.plot(grid_lasso.best_estimator_.coef_, 'o', label="Lasso({})".format(grid_lasso.best_params_['alpha']))
plt.plot(lr.coef_, 'o', label="LR", alpha=.6)
plt.legend()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

scores_lr = cross_val_score(LinearRegression(), X_train_poly, y_train, cv=10)
print("Linear regression poly features: {}".format(scores_lr.mean()))
scores_ridge = cross_val_score(Ridge(), X_train_poly, y_train, cv=10)
print("Ridge regression poly features: {}".format(scores_ridge.mean()))

from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': np.logspace(-3, 3, 7)}
grid = GridSearchCV(Ridge(), param_grid, cv=10)
grid.fit(X_train_poly, y_train)

res = pd.DataFrame(grid.cv_results_)
res.plot("param_alpha", ["mean_train_score", "mean_test_score"], logx=True)
plt.title("Ridge grid search with polynomial features")


print(grid.best_params_, grid.best_score_)
# score with polynomial features is worse!
