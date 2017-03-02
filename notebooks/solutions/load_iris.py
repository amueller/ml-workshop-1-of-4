import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

print("Dataset size: %d  number of features: %d  number of classes: %d"
      % (X.shape[0], X.shape[1], len(np.unique(y))))

X_train, X_test, y_train, y_test = train_test_split(X, y)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.figure()
plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train)
