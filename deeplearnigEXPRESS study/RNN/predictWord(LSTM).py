import numpy as np
from tensorflow.keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

text_data =  """Soft as the voice of an angel\n
                breating a lesson unhead\n
                Hope with a gentle persuasion\n
                Whispers her comforting word\n
                Wait till the darkness is over\n
                Wait till the tempest is donw\n
                Hope for sunshine tomorrow\n
                After the shower
            """

#텍스트 정수 변환
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
encoded = tokenizer.texts_to_sequences([text_data])[0]  #정수 시퀀스로 변환

#어휘 크기 알아내기
vocab_size = len(tokenizer.word_index) + 1
print("어휘 크기: %d" %vocab_size)

#순환 신경망을 위한 단어 시퀀스 생성
sequences = list()
for i in range(1, len(encoded)):
    sequence = encoded[i-1:i+1]
    sequences.append(sequence)

#훈련 데이터와 정답 생성
sequences = np.array(sequences)
X,y = sequences[:,0], sequences[:,1]

#모델 생성
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))

#모델 학습
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=2)

#테스트
#테스트 단어 정수 인코딩
test_text = "Wait"
enncoded = tokenizer.texts_to_sequences([test_text])[0]
encoded = np.array(encoded)

#가장 높은 예측 유닛을 찾음
onehot_output = model.predict(encoded)
output = np.argmax(onehot_output)
print(output)

#다음 단어를 출력
print(test_text, "=>", end=" ")
for word, index in tokenizer.word_index.items():
    if index==output:
        print(word)