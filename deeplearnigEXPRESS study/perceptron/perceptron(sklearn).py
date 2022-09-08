from sklearn.linear_model import Perceptron

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 0, 0, 1]

clf = Perceptron(tol=1e-3, random_state=0) #tol: 종료 조건
clf.fit(X,y)

print(clf.predict(X))
