#혼공머신 교재 4-1챕터 

import pandas as pd
import numpy as np
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


#로지스틱 회귀(logistic regression, 분류 문제)

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]  #bream과 smelt만 분리

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)