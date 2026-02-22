import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            predictions.append(Counter(k_labels).most_common(1)[0][0])
        return predictions


# Example
X = np.array([[1,2],[2,3],[3,3],[6,5],[7,7]])
y = np.array([0,0,0,1,1])

model = KNN(k=3)
model.fit(X,y)

print("kNN Prediction:", model.predict(np.array([[5,5]])))
