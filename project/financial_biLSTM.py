import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from konlpy.tag import Okt
import pickle


data = pd.read_csv('./project/new_list_result.csv', encoding='cp949')
print(len(data))
# print(data[:10])
# print(data['headline'].nunique())
data.drop_duplicates(subset=['headline'], inplace=True)
print(len(data))
# print(data.isnull().sum())
data = data.dropna(how = 'any')
# print(data.isnull().sum())
train_data, test_data = train_test_split(data, test_size=0.1)
# print(len(train_data))
# print(len(test_data))
# data['result'].value_counts().plot(kind='bar')
# plt.show()
# print(train_data[:10])
train_data['headline']=train_data['headline'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
# print(train_data[:10])
train_data['headline'] = train_data['headline'].str.replace('^ +', "")
train_data['headline'].replace('', np.nan, inplace=True)
# print(train_data.isnull().sum())
train_data = train_data.dropna(how = 'any')
# print(len(train_data))

test_data['headline']=test_data['headline'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
test_data['headline'] = test_data['headline'].str.replace('^ +', "")
test_data['headline'].replace('', np.nan, inplace=True)
# print(test_data.isnull().sum())
test_data = test_data.dropna(how = 'any')
# print(len(test_data))

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
# print(train_data[:3])
okt = Okt()

X_train = []
X_test = []

# for sentence in tqdm(train_data['headline']):
#     tokenized_sentence = okt.morphs(sentence, stem=True)
#     stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
#     X_train.append(stopwords_removed_sentence)
# with open('X_train.pkl', 'wb') as f:
#     pickle.dump(X_train, f)
# # print(X_train[:3])

# for sentence in tqdm(test_data['headline']):
#     tokenized_sentence = okt.morphs(sentence, stem=True)
#     stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
#     X_test.append(stopwords_removed_sentence)
# with open('X_test.pkl', 'wb') as f:
#     pickle.dump(X_test, f)

with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)  
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
# print(tokenizer.word_index)
# print(len(tokenizer.word_index))

threshold = 2
total_cnt = len(tokenizer.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# print('단어 집합(vocabulary)의 크기 :',total_cnt)
# print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
# print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
# print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 1

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# print(X_train[:3])

y_train = np.array(train_data['result'])
y_test = np.array(test_data['result'])

# print(len(X_train))
# print(len(y_train))

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
# print(len(X_train))
# print(len(y_train))

# print('최대 길이 :',max(len(headline) for headline in X_train))
# print('평균 길이 :',sum(map(len, X_train))/len(X_train))
# plt.hist([len(headline) for headline in X_train], bins=50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()
def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
#   print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 20
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

import re
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 256

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('./model/financial_best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=100, callbacks=[es, mc], batch_size=256, validation_split=0.1)
