import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

lines = pd.read_csv('./data/fra.txt', names=['src','tar','lic'],sep='\t')
del lines['lic']
# print(len(lines))

lines = lines.loc[:,'src':'tar']
lines = lines[0:60000]
# print(lines.sample(10))

lines.tar = lines.tar.apply(lambda x : '\t' + x + '\n')
# print(lines.sample(10))

src_vocab = set()
for line in lines.src:
    for char in line:
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
# print(len(src_vocab)+1)
# print(len(tar_vocab)+1)

src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
# print(src_vocab[45:75])
# print(tar_vocab[45:75])

src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
# print(src_to_index)
# print(tar_to_index)

encoder_input = []

for line in lines.src:
    encoded_line = []
    for char in line:
        encoded_line.append(src_to_index[char])
    encoder_input.append(encoded_line)
# print(encoder_input[:5])

decoder_input = []
for line in lines.tar:
    encoded_line = []
    for char in line:
        encoded_line.append(tar_to_index[char])
    decoder_input.append(encoded_line)
# print(decoder_input[:5])

decoder_target = []
for line in lines.tar:
  timestep = 0
  encoded_line = []
  for char in line:
    if timestep > 0:
      encoded_line.append(tar_to_index[char])
    timestep = timestep + 1
  decoder_target.append(encoded_line)
# print('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])

max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
print(max_src_len)
print(max_tar_len)

encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np

encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)
decoder_outputs,_,_=decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_softmax__layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax__layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64, epochs=40, validation_split=0.2)
