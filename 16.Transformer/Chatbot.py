import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import Trans_Model as trans

# urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="./data/ChatBotData.csv")
train_data = pd.read_csv('./data/ChatBotData.csv')
# print(train_data.head())
# print(len(train_data))
# print(train_data.isnull().sum())

##      구두점 사이에 공백 집어넣기
questions = []
for sentence in train_data['Q']:
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['A']:
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    sentence.strip()
    answers.append(sentence)
# print(questions[:5])
# print(answers[:5])

##      서브워드텍스트 인코더
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size = 2**13
)

##      시작토큰(SOS)과 종료토큰(EOS)에 정수 부여
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size+1]
VOCAB_SIZE = tokenizer.vocab_size+2

# print('시작 토큰 번호 :',START_TOKEN)
# print('종료 토큰 번호 :',END_TOKEN)
# print('단어 집합의 크기 :',VOCAB_SIZE)

##      정수 인코딩 및 디코딩
# 서브워드텍스트인코더 토크나이저의 .encode()를 사용하여 텍스트 시퀀스를 정수 시퀀스로 변환.
# print('임의의 질문 샘플을 정수 인코딩 : {}'.format(tokenizer.encode(questions[20])))

# # .encode() , .decode() 테스트
# sample_string = questions[20]
# tokenized_string = tokenizer.encode(sample_string)
# print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))
# original_string = tokenizer.decode(tokenized_string)
# print ('기존 문장: {}'.format(original_string))
# for ts in tokenized_string:
#     print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))

MAX_LENGTH = 40
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []
    for (sentence1, sentence2) in zip(inputs, outputs):
        #   시작, 종료 토큰 추가
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)
    #   패딩
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post'
    )
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post'
    )

    return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter(questions, answers)
# print('질문 데이터의 크기(shape) :', questions.shape)
# print('답변 데이터의 크기(shape) :', answers.shape)

# print(questions[20])
# print(answers[20])

##      데이터셋 구성하기
#   교사강요를 위해 디코더의 입력과 실제값 시퀀스를 구성한다.
BATCH_SIZE = 64
BUFFER_SIZE = 200000
#   디코더의 실제값 시퀀스에서는 시작 토큰을 제거한다.
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs':questions,
        'dec_inputs':answers[:,:-1]
    },
    {
        'outputs':answers[:,1:]
    },
))
dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dadtaset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

##      트랜스포머 만들기
tf.keras.backend.clear_session()
# transformer = trans.transformer()
# CustomSchedule = trans.CustomSchedule()
# loss_function = trans.loss_function()

#   하이퍼파라미터
D_MODEL = 256
NUM_LAYERS = 2
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

model = trans.transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
)
learning_rate = trans.CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
  # 레이블의 크기는 (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=trans.loss_function, metrics=[accuracy])

EPOCHS = 3
model.fit(dataset, epochs=EPOCHS)

model.save('./model/chatbot.keras')