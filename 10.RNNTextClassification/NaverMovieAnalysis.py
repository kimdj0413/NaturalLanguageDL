import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('./data/ratings_train.txt')
test_data = pd.read_table('./data/ratings_test.txt')

# print(len(train_data))
# print(train_data[:5])

##      중복제외 개수
# print(train_data['document'].nunique(), train_data['label'].nunique())
train_data.drop_duplicates(subset=['document'], inplace=True)
# print(len(train_data))
train_data['label'].value_counts().plot(kind='bar')
# plt.show()
# print(train_data.groupby('label').size().reset_index(name = 'count'))
# print(train_data.isnull().values.any())
# print(train_data.isnull().sum())
# print(train_data.loc[train_data.document.isnull()])

train_data = train_data.dropna(how='any')   # null 값 행 제거
# print(train_data.isnull().values.any())
# print(train_data[:5])
# 한글과 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
# print(train_data[:5])
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)
# print(train_data.isnull().sum())
train_data = train_data.dropna(how='any')
# print(len(train_data))

test_data.drop_duplicates(subset = ['document'], inplace=True) 
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
test_data['document'] = test_data['document'].str.replace('^ +', "")
test_data['document'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how='any')
# print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
# okt = Okt()
# X_train = []
# for sentence in tqdm(train_data['document']):
#     tokenized_sentence = okt.morphs(sentence, stem=True)
#     stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
#     X_train.append(stopwords_removed_sentence)

# X_test = []
# for sentence in tqdm(test_data['document']):
#     tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
#     stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
#     X_test.append(stopwords_removed_sentence)

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X_train)
# print(tokenizer.word_index)

##      빈도수가 3 이하인 단어 비중
# threshold = 3
# total_cnt = len(tokenizer.word_index)
# rare_cnt = 0
# total_freq = 0
# rare_freq = 0

# for key, value in tokenizer.word_counts.items():
#     total_freq = total_freq + value

#     if(value < threshold):
#         rare_cnt = rare_cnt + 1
#         rare_freq = rare_freq + value

# print('단어 집합(vocabulary)의 크기 :',total_cnt)
# print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
# print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
# print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# vocab_size = total_cnt - rare_cnt + 1

# tokenizer = Tokenizer(vocab_size)
# tokenizer.fit_on_texts(X_train)
# X_train = tokenizer.texts_to_sequences(X_train)
# X_test = tokenizer.texts_to_sequences(X_test)

# y_train = np.array(train_data['label'])
# y_test = np.array(test_data['label'])

# drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

# X_train = np.delete(X_train, drop_train, axis=0)
# y_train = np.delete(y_train, drop_train, axis=0)

# print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
# print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
# plt.hist([len(review) for review in X_train], bins=50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()

max_len = 30

# X_train = pad_sequences(X_train, maxlen=max_len)
# X_test = pad_sequences(X_test, maxlen=max_len)

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim))
# model.add(LSTM(hidden_units))
# model.add(Dense(1, activation='sigmoid'))

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('./model/best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
sentiment_predict('감독 뭐하는 놈이냐?')
