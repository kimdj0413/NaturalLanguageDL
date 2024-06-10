import pandas as pd
import numpy as np
from string import punctuation

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

##      데이터 로드 및 unknown 삭제
df = pd.read_csv('ArticlesApril2018.csv')
# print(df.head())
# print(df.columns)
# print(df['headline'].isnull().values.any())
# print(df['headline'][1])
headline = []
headline.extend(list(df.headline.values))
# print(headline[:5])
# print(len(headline))
headline = [word for word in headline if word != "Unknown"]
# print(len(headline))

##      구두점 제거 및 소문자화
def repreprocessing(raw_sentences):
    preprocessed_sentence = raw_sentences.encode("utf8").decode("ascii",'ignore')
    return ''.join(word for word in preprocessed_sentence if word not in punctuation).lower()

preprocessed_headline = [repreprocessing(x) for x in headline]
# print(preprocessed_headline[:5])

##      토큰화
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_headline)
vocab_size = len(tokenizer.word_index)+1
# print(vocab_size)

##      정수 인코딩 및 하나의 문장을 여러줄로 분해
sequences = list()

for sentence in preprocessed_headline:
    encoded = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

# print(sequences[:11])

##      인덱스를 단어로 바꿔보기
index_to_word = {}
for key, value in tokenizer.word_index.items():
    index_to_word[value] = key
# print('빈도수 상위 582번 단어 : {}'.format(index_to_word[582]))

##      패딩 작업 및 레이블 분류
max_len = max(len(l) for l in sequences)
# print(max_len)

sequences = pad_sequences(sequences, maxlen = max_len, padding='pre')
# print(sequences[:3])
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
# print(y[:3])

##      원-핫 인코딩
y = to_categorical(y, num_classes = vocab_size)
# print(y[:5])

##      모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

embedding_dim = 10
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)

##      문장 생성 함수
def sentence_generation(model, tokenizer, current_word, n):
    init_word = current_word
    sentence = ''

    for _ in range(n):
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen = max_len-1, padding='pre')

        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        for word, index in tokenizer.word_index.items():
            if index == result:
                break

        current_word = current_word+' ' +word

        sentence = sentence+ ' ' + word
    
    sentence = init_word + sentence
    return sentence

print(sentence_generation(model, tokenizer, 'i', 10))