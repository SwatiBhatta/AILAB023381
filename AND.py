import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, epochs=10):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                linear = np.dot(X[i], self.w) + self.b
                y_pred = 1 if linear >= 0 else 0
                error = y[i] - y_pred

                self.w += self.lr * error * X[i]
                self.b += self.lr * error

    def predict(self, X):
        linear = np.dot(X, self.w) + self.b
        return np.where(linear >= 0, 1, 0)


# AND Gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

p = Perceptron()
p.fit(X,y)

print("AND Predictions:", p.predict(X))
