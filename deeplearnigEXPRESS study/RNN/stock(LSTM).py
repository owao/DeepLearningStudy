import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

samsung = fdr.DataReader('005930', '2016')
openValues = samsung[['Open']]

#특징 정규화(0~1 사이의 값)
scaler = MinMaxScaler(feature_range = (0,1))
scaled = scaler.fit_transform(openValues)

#train / test 분리
TEST_SIZE = 200
train_data = scaled[:-TEST_SIZE]
test_data = scaled[-TEST_SIZE:]

#순차 훈련 데이터 생성기
def make_sample(data, window):
    train=[]
    target=[]
    for i in range(len(data)-window):
        train.append(data[i:i+window])
        target.append(data[i+window])
    return np.array(train), np.array(target)

X_train, y_train = make_sample(train_data,30)
X_test, y_test = make_sample(test_data, 30)

#LSTM 구축
model = Sequential()
model.add(LSTM(16,
               input_shape=(X_train.shape[1],1),
               activation='tanh',
               return_sequences=False
               )
        )
model.add(Dense(1))

#모델 학습
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=16)

#예측
pred=model.predict(X_test)

#그래프
plt.figure(figsize=(12,9))
plt.plot(y_test, label='stock price')
plt.plot(pred, label='predicted stock price')
plt.legend()
plt.show()