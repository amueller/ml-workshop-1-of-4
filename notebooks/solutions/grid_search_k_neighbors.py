from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors': [1, 3, 5, 7, 10]}

grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid)
grid.fit(X_train, y_train)
print("best parameters: %s" % grid.best_params_)
print("Training set accuracy: %s" % grid.score(X_train, y_train))
print("Test set accuracy: %s" % grid.score(X_test, y_test))
