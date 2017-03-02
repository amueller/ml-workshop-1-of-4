from sklearn.neighbors import KNeighborsClassifier
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
param_grid = {'kneighborsclassifier__n_neighbors': [1, 3, 5, 10]}
grid = GridSearchCV(pipe, param_grid)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.score(X_test, y_test))
