#train 데이터 전처리(ID 삭제, 마침표 삭제, 대문자->소문자(물음표 삭제 x))
#test 데이터 전처리(ID 삭제, 마침표 삭제, 대문자->소문자(물음표 삭제 x))

#train 데이터(정수 인코딩)를 dialogue_id별로 분리해서 저장
#dialogue_id별로 단어 시퀀스 생성
#단어 시퀀스 훈련(LSTM)

#test 데이터(정수 인코딩)를 dialogue_id별로 분리해서 저장
#dialogue_id별로 단어 시퀀스 생성
#단어 시퀀스 훈련(LSTM)

#submission에 저장

import numpy as np
import pandas as pd
import re
import nltk
import gensim
import torch
from gensim.models.word2vec import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import imdb


# road csv datas
# train: 학습용 데이터 / test: 실제 데이터 / submission: 제출용 label
train = pd.read_csv("C:/Dacon__/dataset/train.csv",)
test = pd.read_csv("C:/Dacon__/dataset/test.csv")
submission = pd.read_csv("C:/Dacon__/dataset/sample_submission.csv")
print(train)


#train, test 데이터 전처리(ID/Speaker 삭제, 마침표 삭제, 대문자->소문자(물음표 삭제 x))
train.info() #ID, Utterance, Speaker, Dialogue_ID, Target / 9989 entries
train=train.drop(['ID', 'Speaker'], axis=1)
test=test.drop(['ID', 'Speaker'], axis=1)

print(train['Target'].unique()) #neutral, surprise, fear, sadness, joy, disgust, anger
train.Target.replace(['neutral'], 0, inplace=True)
train.Target.replace(['surprise'], 1, inplace=True)
train.Target.replace(['fear'], 2, inplace=True)
train.Target.replace(['sadness'], 3, inplace=True)
train.Target.replace(['joy'], 4, inplace=True)
train.Target.replace(['disgust'], 5, inplace=True)
train.Target.replace(['anger'], 6, inplace=True)


#nltk.download('stopwords')
#nltk.download('punkt')

#word2vec 모델 생성
sentences = train.Utterance
model = Word2Vec(sentences, min_count = 1, window = 3)


#train 전처리
for i in range(9989):
    #영어만 남기기
    a = re.sub('[^a-zA-Z]', ' ', train.Utterance[i])
    #대문자 제거
    b = a.lower().split()
    #불용어 제거
    stops = set(stopwords.words('english'))
    c = [word for word in b if not word in stops]
    #어간 추출
    stemmer = nltk.stem.SnowballStemmer('english')
    d = [stemmer.stem(word) for word in c]
    #최종 문자열
    e = ' '.join(d)
    train.Utterance[i] = e
    #텍스트 벡터화
    list=[]
    for j in range(len(train.Utterance[i])):
        list.append(model.wv[train.Utterance[i][j]])
    vector = (np.array([sum(x) for x in zip(*list)])) / len(list)
    train.Utterance[i] = vector #vector size:100
    #빈 Utterance를 영행렬로 채우기
    if vector.size == 0:
        train.Utterance[i] = np.zeros(100)
print(train)


#test 전처리
for i in range(2610):
    #영어만 남기기
    a = re.sub('[^a-zA-Z]', ' ', test.Utterance[i])
    #대문자 제거
    b = a.lower().split()
    #불용어 제거
    stops = set(stopwords.words('english'))
    c = [word for word in b if not word in stops]
    #어간 추출
    stemmer = nltk.stem.SnowballStemmer('english')
    d = [stemmer.stem(word) for word in c]
    #최종 문자열
    e = ' '.join(d)
    test.Utterance[i] = e
    #텍스트 벡터화
    list=[]
    for j in range(len(test.Utterance[i])):
        list.append(model.wv[test.Utterance[i][j]])
    vector = (np.array([sum(x) for x in zip(*list)])) / len(list)
    test.Utterance[i] = vector
    #빈 Utterance를 영행렬로 채우기
    if vector.size == 0:
        test.Utterance[i] = np.zeros(100)
print(test)



#train 데이터(정수 인코딩)를 dialogue_id별로 분리해서 저장
dial_id = 0
train_dial = []
list_ = []
for i in range(9989):
    if train.Dialogue_ID[i] == dial_id:
        list_.append(train.Utterance[i])
    else:
        dial_id += 1
        train_dial.append(list_)
        list_ = []


#dialogue_id별로 문장 시퀀스 생성
def make_sample(data,  window):
    train_data = []
    for i in range(len(data)-window):
        train_data.append(data[i:i+window])
    return np.array(train_data)

train_data = []  #(샘플 개수, 2개 문장, 100개 단어)로 모음
for i in range(len(train_dial)):
    X = make_sample(train_dial[i] , 2)
    # print(X.shape,":", i)
    train_data.append(X)
train_data = np.array(train_data)

train_ = []
for i in range(len(train_data)):
    for j in range(len(train_data[i])):
        train_.append(train_data[i][j])

train_data = np.array(train_)

#train label 만들기
train_label = train.iloc[:, 2].values

list_=[]
dial_id = 0
for i in range(9989):
    if train.Dialogue_ID[i] == dial_id:
        list_.append(train_label[i])
    else:
        dial_id += 1
        i += 2
train_label = np.array(list_)

print(train_data.shape)
print(train_label.shape)


#모델 생성
model = Sequential()
model.add(LSTM(100,
               activation='tanh',
               return_sequences=False))
model.add(Dense(7, activation='relu'))

#모델 학습
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_label[:7060], epochs=100, verbose=2)


#test 데이터(정수 인코딩)를 dialogue_id별로 분리해서 저장
#dialogue_id별로 단어 시퀀스 생성
#단어 시퀀스 훈련(LSTM)

#submission에 저장