###################
###     RNN     ###
###################

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

text = """경마장에 있는 말이 뛰고 있다\n
그의 말이 법이다\n
가는 말이 고와야 오는 말이 곱다\n"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index)+1
# print("단어집합의 크기: %d" %vocab_size)
# print(tokenizer.word_index)

##      훈련데이터 만들기
sequences = list()
for line in text.split('\n'):
    encoded = tokenizer.texts_to_sequences([line])[0]
    # print(encoded)
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        # print(sequence)
        sequences.append(sequence)
# print(len(sequences))

##      패딩
max_len = max(len(l) for l in sequences)
# print('샘플 최대 길이 : {}'.format(max_len))
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
# print(sequences)

##      마지막 단어를 레이블로 분류
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
# print(X)
# print(y)

##      레이블 값 원핫인코딩
y = to_categorical(y, num_classes=vocab_size)
# print(y)

##      모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN

embedding_dim = 10
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X,y,epochs=200,verbose=2)