import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

print("Dataset size: %d  number of features: %d  number of classes: %d"
      % (X.shape[0], X.shape[1], len(np.unique(y))))

X_train, X_test, y_train, y_test = train_test_split(X, y)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.figure()
plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])

import sklearn.datasets
import os
import pandas as pd
iris_path = os.path.join(sklearn.datasets.__path__[0], 'data', 'iris.csv')
iris_df =  pd.read_csv(iris_path, header=None)
display(iris_df.head())

iris_df = pd.read_csv(iris_path, skiprows=1, header=None)
display(iris_df.head())

features = iris_df.iloc[:, :4]
target = iris_df.iloc[:, 4]