import numpy as np
epsilon = 0.0000001

def perceptron1(x1, x2):  #AND 연산 파이썬 사용해 구현
    w1, w2, b = 1.0, 1.0, -1.5
    sum = x1*w1 + x2*w2 + b
    if sum > epsilon:
        return 1
    else:
        return 0

def perceptron2(x1, x2):  #AND 연산 넘파이 사용해 구현(벡터의 내적 사용)
    X = np.array([x1, x2])
    W = np.array([1.0, 1.0])
    B = -1.5
    sum = np.dot(W, X) + B
    if sum > epsilon:
        return 1
    else:
        return 0
