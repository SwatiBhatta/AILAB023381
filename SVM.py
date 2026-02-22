from sklearn import svm

X = [[1,2],[2,3],[3,3],[6,5],[7,7]]
y = [0,0,0,1,1]

model = svm.SVC(kernel='linear')
model.fit(X,y)

print("SVM Prediction:", model.predict([[5,5]]))
