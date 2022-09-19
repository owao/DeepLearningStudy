import matplotlib.pyplot as plt
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#모델 세팅
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512,activation='relu', input_shape=(784,)))  # 28*28=784
model.add(tf.keras.layers.Dense(10,activation='sigmoid'))

#컴파일 단계
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

#데이터 전처리
train_images = train_images.reshape((60000,784))
train_images = train_images.astype('float32') / 255.0
test_images = test_images.reshape((10000,784))
test_images = test_images.astype('float32') / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

#모델 학습
model.fit(train_images, train_labels, epochs=5, batch_size=128)

#테스트
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('테스트 정확도: ', test_acc)