from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV

pipeline = make_pipeline(CountVectorizer(),
                         LinearSVC())
pipeline.fit(text_train, y_train)
print("Pipeline test score: %f"
      % pipeline.score(text_test, y_test))
visualize_coefficients(pipeline.named_steps['linearsvc'],
                       pipeline.named_steps['countvectorizer'].get_feature_names())

param_grid = {'linearsvc__C': 10. ** np.arange(-3, 3)}

grid_search = GridSearchCV(pipeline, param_grid=param_grid)
grid_search.fit(text_train, y_train)

print("best parameters : %s" % grid_search.best_params_)
print("Grid-searched test score: %f"
      % grid_search.score(text_test, y_test))

est = grid_search.best_estimator_
visualize_coefficients(est.named_steps['linearsvc'],
                       est.named_steps['countvectorizer'].get_feature_names())

param_grid = {'linearsvc__C': 10. ** np.arange(-3, 3),
              "countvectorizer__ngram_range": [(1, 1), (1, 2), (2, 2)]}
grid_search = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=3)
grid_search.fit(text_train, y_train)

print("best parameters with n-gram search: %s" % grid_search.best_params_)
print("test set score with n-gram search: %s" % grid_search.score(text_test, y_test))

est = grid_search.best_estimator_
visualize_coefficients(est.named_steps['linearsvc'],
                       est.named_steps['countvectorizer'].get_feature_names())
