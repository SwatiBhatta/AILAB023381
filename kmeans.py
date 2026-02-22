import numpy as np

class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.k)]

            for x in X:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(x)

            new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        labels = []
        for x in X:
            distances = [np.linalg.norm(x - c) for c in self.centroids]
            labels.append(np.argmin(distances))
        return labels


# Example
X = np.array([[1,2],[2,3],[3,3],[6,5],[7,7]])

model = KMeans(k=2)
model.fit(X)

print("Cluster Centers:\n", model.centroids)
print("Predictions:", model.predict(X))
