import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')

#평탄화
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

data_train, data_test, target_train, target_test = train_test_split(data, digits.target, test_size=0.2)

#knn 훈련
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(data_train, target_train)

test_pred = knn.predict(data_test)

scores = metrics.accuracy_score(target_test, test_pred)
print(scores)