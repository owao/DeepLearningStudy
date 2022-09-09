import numpy as np
import matplotlib as plt

X = np.array([0.0, 1.0, 2.0])
y = np.array([3.0, 3.5, 5.5])

w = 0   #기울기(가중치)
b = 0   #절편(바이어스)

lrate = 0.01    #학습률
epochs = 1000   #반복 횟수

n = float(len(X))   #입력 데이터의 개수

#경사 하강법(손실 함수 탐색)
for i in range(epochs):
    y_pred = w*X + b
    dw = (2/n) * sum(X * (y_pred - y))  #w에 대해 편미분한 값
    db = (2/n) * sum(y_pred - y)        #b에 대해 편미분한 값
    w = w - lrate*dw
    b = b - lrate*db