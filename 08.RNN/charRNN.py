import numpy as np
import urllib.request
from tensorflow.keras.utils import to_categorical
import re

f = open('alice.txt', 'r', encoding='utf-8')
sentences = []
for sentence in f:  # 한줄씩 읽어드리기
    sentence = sentence.strip()     # whitespace(\r, \n) 제거
    sentence = sentence.lower()     # 소문자화
    sentence = re.sub(r'\s+', ' ', sentence)    # 스페이스바가 두개 이상 나오면 한개로 치환
    sentence = ''.join(char for char in sentence if ord(char) < 128)    # 유티코드 128 밑으로만 남김.

    if len(sentence) > 0:
        sentences.append(sentence)
f.close()

total_data = ' '.join(sentences)    # 하나의 문장으로 통합(문자 사이에 공백)
# print(total_data[:200])

char_vocab = sorted(list(set(total_data)))  # set: 집합으로
vocab_size = len(char_vocab)
# print(vocab_size)

char_to_index = dict((char, index) for index, char in enumerate(char_vocab))
# print(char_to_index)
index_to_char = {}
for key, value in char_to_index.items():
    index_to_char[value] = key

seq_length = 60
n_samples = int(np.floor((len(total_data) - 1) / seq_length))
# print(n_samples)

train_X = []
train_y = []

for i in range(n_samples):
    X_sample = total_data[i * seq_length : (i+1) * seq_length]
    X_encoded = [char_to_index[c] for c in X_sample]
    train_X.append(X_encoded)
    y_sample = total_data[i * seq_length + 1: (i+1) * seq_length + 1]
    y_encoded = [char_to_index[c] for c in y_sample]
    train_y.append(y_encoded)

# print(len(train_X))
# print(len(train_y))
# print(train_X[0])
# print(train_y[0])
# print([index_to_char[i] for i in train_X[0]])
# print([index_to_char[i] for i in train_y[0]])

train_X = to_categorical(train_X)
train_y = to_categorical(train_y)
print(train_X.shape, train_y.shape)
# (2333, 60 ,43) = (n_samples, seq_lenth, vocab_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed

hidden_units = 256

model = Sequential()
model.add(LSTM(hidden_units, input_shape=(None, train_X.shape[2]), return_sequences=True))
model.add(LSTM(hidden_units, return_sequences = True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=80, verbose=0)

def sentence_generation(model, length):
    ix = [np.random.randint(vocab_size)]
    y_char = [index_to_char[ix[-1]]]
    print(ix[-1],'번 문자',y_char[-1],'로 예측을 시작!')
    X = np.zeros((1, length, vocab_size))

    for i in range(length):
        X[0][i][ix[-1]] = 1
        print(index_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(index_to_char[ix[-1]])
    return ('').join(y_char)

result = sentence_generation(model, 100)
print(result)

