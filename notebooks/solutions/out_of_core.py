sgd = SGDClassifier(learning_rate='invscaling', eta0=.5)

for j in range(10):
    for i in range(9):
        X_batch, y_batch = pickle.load(open("data/batch_%02d.pickle" % i, "rb"))
        sgd.partial_fit(X_batch, y_batch, classes=range(10))
    print(sgd.score(X_test, y_test))
