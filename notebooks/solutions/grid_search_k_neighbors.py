from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors': [1, 3, 5, 7, 10]}

grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, return_train_score=True)
grid.fit(X_train, y_train)
print("best parameters: %s" % grid.best_params_)
print("Training set accuracy: %s" % grid.score(X_train, y_train))
print("Test set accuracy: %s" % grid.score(X_test, y_test))
results = grid.cv_results_
plt.plot(param_grid['n_neighbors'], results['mean_train_score'], label="train")
plt.plot(param_grid['n_neighbors'], results['mean_test_score'], label="test")
plt.legend()
