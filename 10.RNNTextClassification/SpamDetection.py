import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('./data/spam.csv', encoding='latin1')
# print(len(data))
# print(data[:5])
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
# print(data[:5])
# print(data.info())
# print(data.isnull().values.any())

##      중복 값 체크
# print(data['v2'].nunique())
data.drop_duplicates(subset=['v2'], inplace=True)
# print(len(data))

##      레이블 값 분포
# data['v1'].value_counts().plot(kind='bar')
# plt.show()
# print(data.groupby('v1').size().reset_index(name='count'))
# print(f'정상 메일의 비율 = {round(data["v1"].value_counts()[0]/len(data) * 100,3)}%')
# print(f'스팸 메일의 비율 = {round(data["v1"].value_counts()[1]/len(data) * 100,3)}%')

X_data = data['v2']
y_data = data['v1']

# print(len(X_data), len(y_data))

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)

tokenizer = Tokenizer() # 빈도가 1회인 단어를 제외 하려면 괄호안에 num_words 사용
tokenizer.fit_on_texts(X_train)
X_train_encoded = tokenizer.texts_to_sequences(X_train)
# print(X_train_encoded[:5])
word_to_index = tokenizer.word_index
# print(word_to_index) # 빈도수가 높은거부터 출력

vocab_size = len(word_to_index) +1
# print(vocab_size)
# print('메일의 최대 길이 : %d' % max(len(sample) for sample in X_train_encoded))
# print('메일의 평균 길이 : %f' % (sum(map(len, X_train_encoded))/len(X_train_encoded)))
# plt.hist([len(sample) for sample in X_data], bins=50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()
max_len=189
X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len)
print(X_train_padded.shape)

from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

embedding_dim = 32
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train_padded, y_train, epochs=4, batch_size=64, validation_split=0.2)

X_test_encoded = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen = max_len)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test_padded, y_test)[1]))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()