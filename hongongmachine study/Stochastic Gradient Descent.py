#혼공머신 교재 4-2챕터 

#확률적 경사 하강법
#손실 함수를 줄이는 방향으로 학습!
#로지스틱 손실 함수: 예측(일반적인 예측값, 음성을 예측했을 때는 1-예측값)과 타깃(1)의 곱의 음수. 작을수록 정확하다!
#로지스틱 손실 함수는 계산한 값을 마이너스 로그변환해서 양수로 전환! 수가 클수록 손실이 크다
#다중 분류에서는 크로스엔트로피 손실 함수, 회귀에는 평균제곱오차나 R^2를 사용


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


#데이터 불러오기&나누기

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)


#데이터 표준화 전처리

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


#확률적 경사 하강법으로 훈련

sc = SGDClassifier(loss='log', max_iter=40, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))


#에포크(전체 시행) 횟수가 어디에서 가장 적합할지 알아보자!

sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)
for _ in range(0,300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))


#찾아낸 에포크의 상태를 그래프로 보자! -> 100번째 쯔음이 훈련세트와 테스트세트가 가장 가깝다

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


#다시 적합한 에포크 값을 지정하고 훈련시키자

sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))