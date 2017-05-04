from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.svm import LinearSVC
iris = load_iris()
X, y = iris.data, iris.target

print(cross_val_score(LinearSVC(), X, y, cv=KFold(len(X), 3)))
print(cross_val_score(LinearSVC(), X, y, cv=StratifiedKFold(y, 3)))
