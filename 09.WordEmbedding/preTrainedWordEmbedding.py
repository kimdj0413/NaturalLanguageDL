import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality', 'bad', 'highly respectable']
y_train = [1, 0, 0, 1, 1, 0, 1]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) +1
# print(vocab_size)
# print(tokenizer.word_index)

X_encoded = tokenizer.texts_to_sequences(sentences)
# print(X_encoded)

max_len = max(len(l) for l in X_encoded)
# print(max_len)

X_train = pad_sequences(X_encoded, maxlen=max_len, padding='post')
y_train = np.array(y_train)
# print(X_train)
# print(y_train)

##      케라스 Embedding()을 직접 사용.(훈련 데이터가 많을 때에만)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, Flatten

# embedding_dim = 4

# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit(X_train, y_train, epochs=100, verbose=2)

##      사전 훈련된 GloVe Embedding 사용
# from urllib.request import urlretrieve, urlopen
# import gzip
# import zipfile

# urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="C:/NaturalLanguage/data/glove.6B.zip")
# zf = zipfile.ZipFile('C:/NaturalLanguage/data/glove.6B.zip')
# zf.extractall() 
# zf.close()

# embedding_dict = dict()
# f = open("C:/NaturalLanguage/data/glove.6B/glove.6B.100d.txt", encoding="utf8")

# for line in f:
#     word_vector = line.split()
#     word = word_vector[0]

#     word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
#     embedding_dict[word] = word_vector_arr
# f.close()

# print(embedding_dict['respectable'])
# # print('벡터의 차원 수 :',len(embedding_dict['respectable']))
# # 벡터 차원수 = 열 사이즈
# embedding_matrix = np.zeros((vocab_size, 100))
# # print(tokenizer.word_index.items())
# # print(embedding_dict['great'])

# for word, index in tokenizer.word_index.items():
#     vector_value = embedding_dict.get(word)
#     if vector_value is not None:
#         embedding_matrix[index] = vector_value

# # print(embedding_matrix[2])

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, Flatten

# output_dim = 100    # 사전 훈련된 워드 임베딩 값 = output_dim

# model = Sequential()
# # 사전 훈련되어 있으므로 trainable = False
# e = Embedding(vocab_size, output_dim, weights=[embedding_matrix], input_length=max_len, trainable=False)
# model.add(e)
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit(X_train, y_train, epochs=100, verbose=2)

##      사전 훈련된 Word2Vec Embedding 사용하기
import gensim

# !pip install gdown
# !gdown https://drive.google.com/uc?id=1Av37IVBQAAntSe1X3MOAl5gvowQzd2_j

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('C:/NaturalLanguage/data/GoogleNews-vectors-negative300.bin.gz', binary=True)
print(word2vec_model.vectors.shape)

embedding_matrix = np.zeros((vocab_size, 300))

def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None
    
for word, index in tokenizer.word_index.items():
    vector_value = get_vector(word)
    if vector_value is not None:
        embedding_matrix[index] = vector_value

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input

model = Sequential()
model.add(Input(shape=(max_len,), dtype='int32'))
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)
