import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#이미지 출력
plt.imshow(train_images[0])

#데이터 정규화(넘파이 배열에 산술연산)
train_images = train_images / 255.0
test_images = test_images / 255.0

#모델 구축
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28,28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#모델 지표 정의(컴파일)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#모델 훈련
model.fit(train_images, train_labels, epochs=5)

#테스트 데이터 확인
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("정확도:", test_acc)