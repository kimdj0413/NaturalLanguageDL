import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters

(X_train, y_train),(X_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

# print('훈련용 뉴스 기사 : {}'.format(len(X_train)))
# print('테스트용 뉴스 기사 : {}'.format(len(X_test)))
num_classes = len(set(y_train)) # 집합으로 중복 제거
# print('카테고리 : {}'.format(num_classes))

# print('첫번째 훈련용 뉴스 기사 :',X_train[0])
# print('첫번째 훈련용 뉴스 기사의 레이블 :',y_train[0])

# print('뉴스 기사의 최대 길이 :{}'.format(max(len(sample) for sample in X_train)))
# print('뉴스 기사의 평균 길이 :{}'.format(sum(map(len, X_train))/len(X_train)))

# plt.hist([len(sample) for sample in X_train], bins=50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()

word_to_index = reuters.get_word_index()
# print(word_to_index)

index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3]=key
# print('빈도수 상위 1번 단어 : {}'.format(index_to_word[4]))

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token

# print(' '.join([index_to_word[index] for index in X_train[0]]))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

vocab_size = 1000
max_len = 100

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=vocab_size, test_split=0.2)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

embedding_dim = 128
hidden_units = 128
num_classes = 46

# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim))
# model.add(LSTM(hidden_units))
# model.add(Dense(num_classes, activation='softmax'))

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# mc = ModelCheckpoint('./model/reuters.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# history = model.fit(X_train, y_train, batch_size=128, epochs=30, callbacks=[es, mc], validation_data=(X_test, y_test))

loaded_model = load_model('./model/reuters.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))