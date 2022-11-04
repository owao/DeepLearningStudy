import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#모델 생성
encoding_dim = 32
input_img = tf.keras.layers.Input(shape=(784,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_img)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.models.Model(input_img, decoded)

#데이터 읽기
mnist = tf.keras.datasets.mnist
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
noise_factor = 0.55

#노이즈 추가
original_train = x_train
original_test = x_test
noise_train = np.random.normal(0, 1, original_train.shape)
noise_test = np.random.normal(0, 1, original_test.shape)
noisy_train = original_train + noise_factor*noise_train
noisy_test = original_test + noise_factor*noise_test

#모델 컴파일 및 학습
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(noisy_train, original_train, epochs=50, batch_size=256, shuffle=True, validation_data=(noisy_test, original_test))

#모델 테스트
denoised_images = autoencoder.predict(noisy_test)
n=5
plt.figure(figsize=(20,6))
for i in range(1, n+1):
    ax = plt.subplot(3,n,i)
    plt.imshow(noisy_test[i].reshape(28,28))
    plt.gray()

    ax = plt.subplot(3,n,i+n)
    plt.imshow(denoised_images[i].reshape(28,28))
    plt.gray()

    ax = plt.subplot(3,n,i+2*n)
    plt.imshow(original_test[i].reshape(28,28))
    plt.gray()
plt.show()