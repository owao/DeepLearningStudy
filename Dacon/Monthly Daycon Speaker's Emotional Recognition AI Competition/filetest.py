import pandas as pd

import checkPreference
import checkData

# road csv datas
# train: 학습용 데이터 / test: 실제 데이터 / submission: 제출용 label
train = pd.read_csv("C:/Dacon__/dataset/train.csv")
test = pd.read_csv("C:/Dacon__/dataset/test.csv")
submission = pd.read_csv("C:/Dacon__/dataset/sample_submission.csv")

# check the preference
checkPreference.route()
checkPreference.csvRoad(train, 5)
checkPreference.csvRoad(test, 5)

# check the shape of data
checkData.shape(train)
checkData.shape(test)
checkData.nameofcolumn(train)
checkData.nameofcolumn(test)

# check missing value
print("count missing value of train data\n")
checkData.missingValue(train)
print("\ncount missing value of test data\n")
checkData.missingValue(test)
