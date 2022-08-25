#혼공머신 교재 3-1챕터

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error


#제대로 된 농어 데이터

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])


#그래프로 나타내보자

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


#훈련용 데이터와 테스트 데이터로 분리한 다음 사이킷런에 맞게 열이 2인 행렬로 바꿔준다

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1, 1)  #42개 데이터. -1은 전체 사이즈대로 행을 유지한다는 뜻
test_input = test_input.reshape(-1, 1)  #14개 데이터.


#모델 훈련
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print("test data score: ",knr.score(test_input, test_target))


#예측 오찻값 확인
test_prediction = knr.predict(test_input)  #테스트 세트 예측
mae = mean_absolute_error(test_target, test_prediction)  #평균 절댓값 오차를 계산(예측이 얼마나 빗나갔는지 보기)
print("error: ", mae)


#과대(과소)적합 확인
print("test score: ",knr.score(test_input, test_target))
print("train score: ",knr.score(train_input, train_target))
if (knr.score(test_input, test_target)>knr.score(train_input, train_target)):
    print("과소적합입니다.(테스트 데이터 점수가 더 높거나 둘 다 점수가 지나치게 낮음)")
elif (knr.score(test_input, test_target)<knr.score(train_input, train_target)):
    print("과대적합입니다.(훈련 데이터 점수가 더 높음)")


#과대(과소)적합 조정
knr.n_neighbors = 3  #이웃의 갯수를 5->3으로 설정 
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))