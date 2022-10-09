import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#픽셀 정규화
train_images, test_images = train_images/255.0, test_images/255.0

#컨벌루션 신경망 생성
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


print(model.summary())


#모델 훈련
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

#테스트
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('테스트 정확도: ', test_acc)