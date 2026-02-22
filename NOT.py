X = np.array([[0],[1]])
y = np.array([1,0])

p = Perceptron()
p.fit(X,y)

print("NOT Predictions:", p.predict(X))
