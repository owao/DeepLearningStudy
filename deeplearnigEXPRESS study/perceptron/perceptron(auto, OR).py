import numpy as np
epsilon = 0.0000001 #부동소수점 오차 방지

def step_func(t):   #활성화 함수
    if t > epsilon: return 1
    else: return 0

X = np.array([  #훈련 데이터셋
    [0, 0, 1],   #맨 끝의 1은 바이어스를 위한 입력 신호 1
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
    ])

y = np.array([0, 1, 1, 1])  #정답 행렬
W = np.zeros(len(X[0]))     #가중치 저장 행렬(global로 이곳저곳에서 불러서 씀)

def perceptron_fit(X, Y, epochs=10):
    global W
    eta = 0.2       #학습률
    for t in range(epochs):
        print("epoch=", t, "=================")
        for i in range(len(X)):
            predict = step_func(np.dot(X[i],W))
            error = Y[i] - predict          #오차 계산
            W += eta * error * X[i]         #가중치 업데이트
            print("현재 처리 입력=", X[i], "정답=", Y[i], "출력=", predict, "변경된 가중치=", W)
        print("==========================")

def perceptron_predict(X,Y):
    global W
    for x in X:
        print(x[0], x[1], "->", step_func(np.dot(x,W)))

perceptron_fit(X,y,4)
perceptron_predict(X,y)
