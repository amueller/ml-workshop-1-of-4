from sklearn.manifold import TSNE
tsne = TSNE()
X_tsne = tsne.fit_transform(X)
plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
