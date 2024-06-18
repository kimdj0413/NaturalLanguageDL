import os
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf
import unicodedata
import urllib3
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sent):
    if re.search(r'[a-zA-Z]',sent):
        sent = to_ascii(sent.lower())
    sent = re.sub(r"([?.!,¿])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z가-힣!.?]+", r" ", sent)
    sent = re.sub(r"\s+", " ", sent)
    return sent

# en_sent = u"Have you had dinner?"
# kor_sent = u"혹시, 저녁 먹었니?"

# print(preprocess_sentence(en_sent))
# print(preprocess_sentence(kor_sent))

def load_preprocessed_data():
    encoder_input, decoder_input, decoder_target = [],[],[]

    with open("./data/kor.txt","r",encoding='utf-8') as lines:
        for i, line in enumerate(lines):
            src_line, tar_line, _ = line.strip().split('\t')
            # print(src_line)
            src_line = [w for w in preprocess_sentence(src_line).split()]
            tar_line = preprocess_sentence(tar_line)
            tar_line_in = [w for w in ("<sos> "+tar_line).split()]
            tar_line_out = [w for w in (tar_line+" <eos>").split()]

            encoder_input.append(src_line)
            decoder_input.append(tar_line_in)
            decoder_target.append(tar_line_out)
    return encoder_input, decoder_input, decoder_target

sents_en_in, sents_kor_in, sents_kor_out = load_preprocessed_data()
# print('인코더의 입력 :',sents_en_in[:5])
# print('디코더의 입력 :',sents_kor_in[:5])
# print('디코더의 레이블 :',sents_kor_out[:5])

tokenizer_en = Tokenizer(filters="", lower=False)
tokenizer_en.fit_on_texts(sents_en_in)
encoder_input = tokenizer_en.texts_to_sequences(sents_en_in)
encoder_input = pad_sequences(encoder_input, padding='post')

tokenizer_kor = Tokenizer(filters="", lower=False)
tokenizer_kor.fit_on_texts(sents_kor_in)
tokenizer_kor.fit_on_texts(sents_kor_out)

decoder_input = tokenizer_kor.texts_to_sequences(sents_kor_in)
decoder_input = pad_sequences(decoder_input, padding="post")

decoder_target = tokenizer_kor.texts_to_sequences(sents_kor_out)
decoder_target = pad_sequences(decoder_target, padding='post')

# print(encoder_input.shape)
# print(decoder_input.shape)
# print(decoder_target.shape)

src_vocab_size = len(tokenizer_en.word_index)+1
tar_vocab_size = len(tokenizer_kor.word_index)+1
# print(src_vocab_size)
# print(tar_vocab_size)

src_to_index = tokenizer_en.word_index
index_to_src = tokenizer_en.index_word
tar_to_index = tokenizer_kor.word_index
index_to_tar = tokenizer_kor.index_word

indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)
# print(indices)

encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]

# print(encoder_input[5222])
# print(decoder_input[5222])
# print(decoder_target[5222])

n_of_val = int(5900*0.1)

encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]

print('훈련 source 데이터의 크기 :',encoder_input_train.shape)
print('훈련 target 데이터의 크기 :',decoder_input_train.shape)
print('훈련 target 레이블의 크기 :',decoder_target_train.shape)
print('테스트 source 데이터의 크기 :',encoder_input_test.shape)
print('테스트 target 데이터의 크기 :',decoder_input_test.shape)
print('테스트 target 레이블의 크기 :',decoder_target_test.shape)

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
from tensorflow.keras.models import Model
import tensorflow as tf
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# 모든 GPU를 비활성화
# tf.config.set_visible_devices([], 'GPU')
embedding_dim = 64
hidden_units = 64

encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(src_vocab_size, embedding_dim)(encoder_inputs)
enc_masking = Masking(mask_value=0.0)(enc_emb)
encoder_lstm = LSTM(hidden_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_masking)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(tar_vocab_size, hidden_units)
dec_emb = dec_emb_layer(decoder_inputs)
dec_masking = Masking(mask_value=0.0)(dec_emb)

decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_masking, initial_state=encoder_states)

decoder_dense = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')

# model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train,
#           validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
#           batch_size=128, epochs=50)
# model.save('./model/seq2seq.h5')

# 모델 불러오기
model = tf.keras.models.load_model('./model/seq2seq.h5')

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(hidden_units,))
decoder_state_input_c = Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

def decode_sequence(input_seq):
  states_value = encoder_model.predict(input_seq)

  target_seq = np.zeros((1,1))
  target_seq[0, 0] = tar_to_index['<sos>']

  stop_condition = False
  decoded_sentence = ''

  while not stop_condition:
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_char = index_to_tar[sampled_token_index]

    decoded_sentence += ' '+sampled_char

    if (sampled_char == '<eos>' or
        len(decoded_sentence) > 50):
        stop_condition = True

    target_seq = np.zeros((1,1))
    target_seq[0, 0] = sampled_token_index

    states_value = [h, c]

  return decoded_sentence

def seq_to_src(input_seq):
  sentence = ''
  for encoded_word in input_seq:
    if(encoded_word != 0):
      sentence = sentence + index_to_src[encoded_word] + ' '
  return sentence

def seq_to_tar(input_seq):
  sentence = ''
  for encoded_word in input_seq:
    if(encoded_word != 0 and encoded_word != tar_to_index['<sos>'] and encoded_word != tar_to_index['<eos>']):
      sentence = sentence + index_to_tar[encoded_word] + ' '
  return sentence

for seq_index in [3, 50, 100, 300, 1001]:
  input_seq = encoder_input_train[seq_index: seq_index + 1]
  decoded_sentence = decode_sequence(input_seq)

  print("입력문장 :",seq_to_src(encoder_input_train[seq_index]))
  print("정답문장 :",seq_to_tar(decoder_input_train[seq_index]))
  print("번역문장 :",decoded_sentence[1:-5])
  print("-"*50)
