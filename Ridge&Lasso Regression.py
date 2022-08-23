#혼공머신 교재 3-3챕터

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


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

#모델 훈련(1차방정식 선형 회귀)


