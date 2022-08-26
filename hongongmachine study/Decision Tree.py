#혼공머신 교재 5-1챕터

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree


#데이터 불러오기&나누기

wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)  #랜덤스테이트는 실습과 결과를 같게 하려고, 테스트사이즈는 20%가 테스트세트라는 의미


#데이터 표준 전처리

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


#로지스틱 회귀로 예측해보면?

lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
print(lr.coef_, lr.intercept_)  #계수와 절편 확인


#결정 트리(Decision Tree) 사용

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))


#결정 트리의 생김새 보기

plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()


#결정 트리의 간략한 생김새 보기
#gini 계수: 지니 불순도(Gini impurity), 결정 트리는 부모 노드와 자식 노드의 불순도 차이가 크도록 트리를 성장시킴(정보 이득)

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


#과적합을 막고 테스트를 예측을 잘 하기 위한 가지치기!(트리 깊이 지정)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


#결정 트리는 표준화 전처리를 할 필요가 없으므로 전처리 전 세트를 사용해보자

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

plt.figure(figsize=(20,15))  #그림을 알아보기가 더 쉽다!
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


#특성별 중요도 확인

print(dt.feature_importances_)
