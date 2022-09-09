import matplotlib.pylab as plt
from sklearn import linear_model

reg = linear_model.LinearRegression()

X = [[0], [1], [2]]     #데이터는 2차원으로 만들어야 함
y = [3, 3.5, 5.5]

reg.fit(X, y)

#reg.coef_ 는 직선의 기울기
#reg.intercept_ 는 직선의 절편