import numpy as np

#활성화 함수(시그모이드)
def actf(x):
    return 1/(1+np.exp(-x))

#시그모이드를 미분한 함수
def actf_deriv(x):
    return x*(1-x)

#입력 유닛 개수, 은닉층 개수, 출력 유닛 개수, 학습률
inputs, hiddens, outputs = 2, 2, 1
learning_rate = 0.2

#훈련 샘플과 정답
X = np.array([[0,0], [0,1], [1,0], [1,1]])
T = np.array([[0], [1], [1], [0]])

#가중치
W1 = np.array([[0.10], [0.20], [0.30], [0.40]])
W1 = np.reshape(W1, (2, -1))  #predict에서 x와 곱해주기 위해 2행짜리 행렬로 변환
W2 = np.array([[0.50], [0.60]])

#바이어스(단일 값)
B1 = np.array([0.1, 0.2])
B2 = np.array([0.3])

#MLP 순방향 전파 계산 함수
def predict(x):
    layer0 = X
    Z1 = np.dot(layer0, W1)+B1
    layer1 = actf(Z1)
    Z2 = np.dot(layer1, W2)+B2
    layer2 = actf(Z2)
    return layer0, layer1, layer2

def test():
    for x,y in zip(X, T):
        x = np.reshape(x, (1, -1))  #2차원 행렬로 변환(입력해야하므로)
        layer0, layer1, layer2 = predict(x)
        print(x, y, layer2)

test()
