X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

p = Perceptron(epochs=20)
p.fit(X,y)

print("XOR Predictions:", p.predict(X))
