#혼공머신 교재 1-3챕터~2챕터

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#도미 데이터&빙어 데이터 출력

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


#두 데이터를 합하고 사이킷런으로 훈련 세트와 테스트 세트 나누기

fish_data = np.column_stack((bream_length+smelt_length, bream_weight+smelt_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))  #fish_data의 타겟 값(검증 리스트)

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)  #세트 나누기




#머신러닝(K-최근접 이웃 알고리즘)

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)  #모델 학습
kn.score(fish_data, fish_target)

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')  #길이 30, 무게 600인 생선의 위치를 세모로 표시!
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.predict([[30,600]])

'''
fish_data는 모델의 kn._fit_X 변수에, fish_target은 모델의 kn._y변수에 동일하게 저장되어 있음
print(kn._fit_X)
print(kn._y)
'''
