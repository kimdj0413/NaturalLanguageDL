from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

# print('리뷰 최대길이 : {}'.format(max(len(l) for l in X_train)))
# print('리뷰 평균길이 : {}'.format(sum(map(len, X_train))/len(X_train)))
max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

import tensorflow as tf
from tensorflow.keras.layers import Dense

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
    def call(self, values, query):
        hidden_with_time_axis = tf.expand_deims(query,1)
        score = self.V(tf.nn.tanh(self.Wq(values) + self.W2()))