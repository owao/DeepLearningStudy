import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

#학습/테스트 데이터 만들기 함수
def make_sample(data, window):
    train=[]
    target=[]
    for i in range(len(data)-window):
        train.append(data[i:i+window])
        target.append(data[i+window])
    return np.array(train), np.array(target)

#사인파 데이터 생성
seq_data=[]
for i in np.arange(0,1000):
    seq_data += [[np.sin( np.pi * i * 0.01 )]]
X,y = make_sample(seq_data,10)

#RNN 모델
model=Sequential()
model.add(SimpleRNN(10, activation='tanh', input_shape=(10,1)))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer='adam', loss='mse')

#모델 학습
history=model.fit(X, y, epochs=100, verbose=1)
plt.plot(history.history['loss'], label="loss")
plt.show()

#테스트
seq_data=[]
for i in np.arange(0,1000):
    seq_data+=[[np.cos( np.pi*i*0.01 )]]
X,y=make_sample(seq_data,10) #윈도우 크기=10
y_pred=model.predict(X, verbose=0)
plt.plot(np.pi*np.arange(0,990)*0.01, y_pred)
plt.plot(np.pi*np.arange(0,990)*0.01, y)
plt.show()