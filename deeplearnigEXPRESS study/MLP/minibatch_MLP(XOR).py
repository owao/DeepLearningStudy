import numpy as np

#활성화 함수(시그모이드)
def actf(x):
    return 1/(1+np.exp(-x))

#시그모이드를 미분한 함수
def actf_deriv(x):
    return x*(1-x)

#입력 유닛 개수, 은닉층 개수, 출력 유닛 개수, 학습률
inputs, hiddens, outputs = 2, 2, 1
learning_rate = 0.5

#훈련 샘플과 정답
X = np.array([[0,0], [0,1], [1,0], [1,1]])
T = np.array([[0], [1], [1], [0]])

#가중치: -1.0~1.0 사이의 난수로 초기화
W1 = 2*np.random.random((inputs, hiddens))-1
W1 = np.reshape(W1, (2, -1))  #predict에서 x와 곱해주기 위해 2행짜리 행렬로 변환
W2 = 2*np.random.random((inputs, hiddens))-1

#바이어스(단일 값): 0으로 초기화
B1 = np.zeros(hiddens)
B2 = np.zeros(outputs)

#에포크
epoch = 90000

#MLP 순방향 전파 계산 함수
def predict(x):
    layer0 = x
    Z1 = np.dot(layer0, W1)+B1
    layer1 = actf(Z1)
    Z2 = np.dot(layer1, W2)+B2
    layer2 = actf(Z2)
    return layer0, layer1, layer2


#역방향 전파 계산 함수
def fit():
    global W1, W2, B1, B2, epoch
    for i in range(epoch):
        layer0, layer1, layer2 = predict(X)
        layer2_error = layer2 - T
        layer2_delta = layer2_error * actf_deriv(layer2)
        layer1_error = np.dot(layer2_delta, W2.T)
        layer1_delta = layer1_error * actf_deriv(layer1)

        W2 += -learning_rate*np.dot(layer1.T, layer2_delta)/4.0
        W1 += -learning_rate*np.dot(layer0.T, layer1_delta)/4.0
        B2 += -learning_rate*np.sum(layer2_delta, axis=0)/4.0
        B1 += -learning_rate*np.sum(layer1_delta, axis=0)/4.0

def test():
    for x,y in zip(X, T):
        x = np.reshape(x, (1, -1))  #2차원 행렬로 변환(입력해야하므로)
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)

fit()
test()