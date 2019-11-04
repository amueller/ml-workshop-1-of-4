from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Exercise 1, loading data
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Exercise 2
# Training KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("test set score of knn: %f" % knn.score(X_test, y_test))

# Training RandomForest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
rf.score(X_test, y_test)

# Exercise 3

# Perfect classification (accuracy=1) on easy dataset
from sklearn.linear_model import LogisticRegression
X = np.random.uniform(size=(1000, 3))
X[::2] += 1000
y = X[:, 0] > 500
X_train, X_test, y_train, y_test = train_test_split(X, y)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("score on trivial data: ", logreg.score(X_test, y_test))

# Random classification (accuracy=.5) on random data
y = np.random.normal(size=1000) > .0
X_train, X_test, y_train, y_test = train_test_split(X, y)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("score on random data: ", logreg.score(X_test, y_test))
