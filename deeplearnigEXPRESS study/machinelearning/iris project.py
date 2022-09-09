from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

data = iris.data
target = iris.target

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=4)

#최근접 이웃 알고리즘 사용
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(data_train, target_train)

test_pred = knn.predict(data_test)

scores = metrics.accuracy_score(target_test, test_pred)
