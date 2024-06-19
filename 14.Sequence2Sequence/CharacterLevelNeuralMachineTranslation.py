import pandas as pd
import tensorflow as tf
from unidecode import unidecode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

lines = pd.read_csv('./data/fra.txt', names=['src','tar','lic'], sep='\t')
del lines['lic']
# print(len(lines))

lines = lines.loc[:,'src':'tar']
lines = lines[0:60000]
# print(lines.sample(10))     # 랜덤으로 10개만 추출

lines.tar = lines.tar.apply(lambda x : '\t '+x+' \n')
# print(lines.sample(10))

##  문자 집합 구축
src_vocab = set()
for line in lines.src:
    line=unidecode(line)
    for char in line:
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    if line != '\t' or '\n':
        line=unidecode(line)
    # print(line)
    for char in line:
        tar_vocab.add(char)
# print(tar_vocab)
src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
# print('source 문장의 char 집합 :',src_vocab_size)
# print('target 문장의 char 집합 :',tar_vocab_size)

##  정렬 후 인덱싱 처리
src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
# print(src_vocab[45:75])
# print(tar_vocab[45:75])
src_to_index = dict([(word,i+1) for i,word in enumerate(src_vocab)])
tar_to_index = dict([(word,i+1) for i,word in enumerate(tar_vocab)])

# print(src_to_index)
# print(tar_to_index)

##  훈련 데이터 정수 인코딩(입력)
encoder_input = []
for line in lines.src:
    encoded_line = []
    line=unidecode(line)
    for char in line:
        encoded_line.append(src_to_index[char])
    encoder_input.append(encoded_line)
# print('source 문장의 정수 인코딩 :',encoder_input[:5])

##  훈련 데이터 정수 인코딩(출력)
decoder_input = []
for line in lines.tar:
    encoded_line = []
    line=unidecode(line)
    for char in line:
        encoded_line.append(tar_to_index[char])
    decoder_input.append(encoded_line)
# print('target 문장의 정수 인코딩 :',decoder_input[:5])

##  디코더의 실제값에서 시작 심볼 제거
decoder_target = []
for line in lines.tar:
    timestep = 0
    encoded_line = []
    line=unidecode(line)
    for char in line:
        if timestep>0:
            encoded_line.append(tar_to_index[char])
        timestep = timestep+1
    decoder_target.append(encoded_line)
# print('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])

##  패딩작업
max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
# print('source 문장의 최대 길이 :',max_src_len)
# print('target 문장의 최대 길이 :',max_tar_len)
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

##  문자 단위 번역기이므로 워드 임베딩이 아닌 원-핫 인코딩 수행
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

##  decoder_input이 필요한 이유 : 교사 강요를 하기 위해서.

##  훈련 시키기
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

#       인코더
encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True)      #   return_state = True : 내부 상태 전달.

encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

encoder_states = [state_h, state_c]     ##  LSTM에서는 은닉 상태와 셀 상태 두가지를 다음 시점에 전달
#       디코드
decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)    #   return_sequences, return_state = True : 매 시점 출력, 내부 상태 전달

decoder_outputs,_,_ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

# model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64, epochs=40, validation_split=0.2)
# model.save('./model/character.h5')
model = tf.keras.models.load_model('./model/character.h5', compile=False)

##      동작 시키기
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

#   훈련 과정과 달리 LSTM의 리턴하는 은닉상태와 셀 상태를 버리지 않음(교사강요x)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs]+decoder_states_inputs, outputs=[decoder_outputs]+decoder_states)

index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, tar_to_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq]+states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]
        print(sampled_char)

        decoded_sentence+=sampled_char

        if(sampled_char == '\n' or
           len(decoded_sentence) > max_tar_len):
            stop_condition = True
        
        target_seq = np.zeros((1, 1, tar_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]
    return decoded_sentence

for seq_index in [3,50,100,300,1001]: # 입력 문장의 인덱스
  input_seq = encoder_input[seq_index:seq_index+1]
  decoded_sentence = decode_sequence(input_seq)
#   print(decoded_sentence)
  print(35 * "-")
  print('입력 문장:', lines.src[seq_index])
  print('정답 문장:', lines.tar[seq_index][2:len(lines.tar[seq_index])-1])
  print('번역 문장:', decoded_sentence[1:len(decoded_sentence)-1])
