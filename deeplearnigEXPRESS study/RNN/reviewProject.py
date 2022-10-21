import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *

imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000) #몇 개의 단어를 사용할지 지정

x_train = pad_sequences(x_train, maxlen=500)
x_test = pad_sequences(x_test, maxlen=500)


#모델 구축
vocab_size=10000
model = Sequential()
model.add(Embedding(vocab_size,100,input_length=500))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#모델 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_test,y_test))

#모델 평가
results=model.evaluate(x_test, y_test, verbose=2)
print(results)