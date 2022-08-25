#혼공머신 교재 4-1챕터 

import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


#데이터 준비하기

fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(pd.unique(fish['Species']))  #Species feature에서 무슨 값 있는지 보기


#학습 feature와 타깃 feature 분리하기

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()


#train data와 test data 분리하기

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)


#표준화 전처리

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


#k-최근접 이웃 분류기를 응용한 확률 예측

kn = KNeighborsClassifier(n_neighbors = 3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
print(kn.predict(test_scaled[:5]))  #타깃값이 뭐가 가능성이 높은지 순서대로 출력


#로지스틱 회귀(logistic regression, 분류 문제) - 이진 분류 연습

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]  #bream과 smelt만 분리

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))


#로지스틱 회귀 다중 분류(규제를 담당하는 파라미터는 C, 작을수록 규제가 커짐)

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))  #점수 확인
print(lr.score(test_scaled, test_target))


#예측 확률 출력

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))


#다중분류는 시그모이드가 아니라 소프트맥스 함수를 사용해 0~1로 변환한다

decision = lr.decision_function(test_scaled[:5])
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))  #우리 손으로 한 softmax가 앞선 출력값과 같은지 확인