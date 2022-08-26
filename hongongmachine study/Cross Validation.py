#혼공머신 교재 5-2챕터

from curses.ascii import GS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree


#데이터 불러오기&나누기(검증 세트까지)

wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)



#결정 트리(Decision Tree) 사용

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))


#k-fold 교차검증 시도

scores = cross_validate(dt, train_input, train_target)
print(np.mean(scores['test_score']))  #교차검증을 한 score 안에서도 fit time이나 score time이 아닌 실제 점수만 평균!!


#10-fold 교차검증을 하는 코드

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))


#그리드 서치(Grid Search): 최적의 하이퍼파라미터를 찾는다+교차검증까지!

params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)  #그리드 서치 수행
dt = gs.best_estimator_  #최적의 파라미터를 dt에게 저장
print(dt.score(train_input, train_target))
print(gs.best_params_)  #최적의 파라미터 확인


#여러개의 하이퍼파라미터를 그리드 서치로 찾아보자!

params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
         }
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))


#탐색할 파라미터 값을 랜덤 서치(Random Searhch)로 찾는다