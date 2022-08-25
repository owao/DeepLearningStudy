#혼공머신 교재 3-3챕터

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


#pandas를 이용한 데이터 준비

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])


#훈련용 데이터와 테스트 데이터로 분리

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)


#특성 공학(feature engineering): 새 특성 만들기, 사이킷런의 변환기(transfomer) 사용

poly = PolynomialFeatures(include_bias=False) #사이킷런의 선형 모델이 자동으로 절편을 추가할 것이므로 절편 특성 추가는 필요 없음
poly.fit(train_input)
train_poly = poly.transform(train_input)

poly.get_feature_names_out() #특성이 어떻게 만들어졌는지 확인(어떤 입력 조합인지)
test_poly = poly.transform(test_input)


#모델 훈련(다중 회귀)

lr = LinearRegression()
lr.fit(train_poly, train_target)

print("train set:", lr.score(train_poly, train_target))
print("test set:", lr.score(test_poly, test_target))


#더 많은 특성 추가(by feature engineering) but 모델의 과대적합 주의!!

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)


#규제(regularization): 모델이 과대적합되지 않도록 규제하는 것! 선형 회귀에서는 계수나 기울기의 크기를 작게 만드는 일

ss = StandardScaler()  #계수를 규제하기 위해 정규화를 한다
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)



#릿지 회귀(릿지와 라쏘는 선형 회귀 모델에 규제를 추가한 것)

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print("ridge train set:", ridge.score(train_scaled, train_target))
print("ridge test set:", ridge.score(test_scaled, test_target))


#적절한 alpha값(클수록 규제를 심하게 함) 찾기

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       ridge = Ridge(alpha=alpha)
       ridge.fit(train_scaled, train_target)
       train_score.append(ridge.score(train_scaled, train_target))
       test_score.append(ridge.score(test_scaled, test_target))


#alpha값 그래프 그려보기

plt.plot(np.log10(alpha_list), train_score)  #train의 알파값에 따른 점수
plt.plot(np.log10(alpha_list), test_score)  #test의 알파값에 따른 점수
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()


#적합한 alpha 값(0.1)로 ridge 모델 훈련

ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print("ridge train set:", ridge.score(train_scaled, train_target))
print("ridge test set:", ridge.score(test_scaled, test_target))



#라쏘 회귀(릿지와 라쏘는 선형 회귀 모델에 규제를 추가한 것)

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print("lasso train set:", lasso.score(train_scaled, train_target))
print("lasso test set:", lasso.score(test_scaled, test_target))


#적절한 alpha값(클수록 규제를 심하게 함) 찾기

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       lasso = Lasso(alpha=alpha, max_iter=10000) #max_iter: 반복 계산 횟수를 늘리기 위해 지정
       lasso.fit(train_scaled, train_target)
       train_score.append(lasso.score(train_scaled, train_target))
       test_score.append(lasso.score(test_scaled, test_target))


#alpha값 그래프 그려보기

plt.plot(np.log10(alpha_list), train_score)  #train의 알파값에 따른 점수
plt.plot(np.log10(alpha_list), test_score)  #test의 알파값에 따른 점수
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()


#적합한 alpha 값(10)로 ridge 모델 훈련

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print("lasso train set:", lasso.score(train_scaled, train_target))
print("lasso test set:", lasso.score(test_scaled, test_target))


#lasso 모델이 사용하지 않은 특성(안 유용한)이 몇 개인지 살피기
print(np.sum(lasso.coef_ == 0))