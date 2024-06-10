import numpy as np
from tensorflow.keras.utils import to_categorical

raw_text = '''
I get on with life as a programmer,
I like to contemplate beer.
But when I start to daydream,
My mind turns straight to wine.

Do I love wine more than beer?

I like to use words about beer.
But when I stop my talking,
My mind turns straight to wine.

I hate bugs and errors.
But I just think back to wine,
And I'm happy once again.

I like to hang out with programming and deep learning.
But when left alone,
My mind turns straight to wine.
'''
tokens = raw_text.split()       # 단락 제거
raw_text = ' '.join(tokens)
# print(raw_text)

char_vocab = sorted(list(set(raw_text)))
vocab_size = len(char_vocab)
# print(char_vocab)
# print(vocab_size)

length = 11 # 입력 데이터 10 + 예측 데이터 1
sequences = []
for i in range(length, len(raw_text)):
    seq = raw_text[i-length:i]
    sequences.append(seq)
# print(len(sequences))
# print(sequences[:10])

char_to_index = dict((char, index) for index, char in enumerate(char_vocab))

encoded_sequences = []
for sequence in sequences:
    encoded_sequence = [char_to_index[char] for char in sequence]
    encoded_sequences.append(encoded_sequence)
# print(encoded_sequences[:5])

encoded_sequences = np.array(encoded_sequences)

X_data = encoded_sequences[:,:-1]
y_data = encoded_sequences[:,-1]

# print(X_data[:5])
# print(y_data[:5])

X_data_one_hot = [to_categorical(encoded, num_classes=vocab_size) for encoded in X_data]
X_data_one_hot = np.array(X_data_one_hot)
print(X_data_one_hot)
y_data_one_hot = to_categorical(y_data, num_classes=vocab_size)
print(X_data_one_hot.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

hidden_units = 64

model = Sequential()
model.add(LSTM(hidden_units, input_shape=(X_data_one_hot.shape[1], X_data_one_hot.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_data_one_hot, y_data_one_hot, epochs=100, verbose=2)

def sentence_generation(model, char_to_index, seq_length, seed_text, n):

    # 초기 시퀀스
    init_text = seed_text
    sentence = ''

    # 다음 문자 예측은 총 n번만 반복.
    for _ in range(n):
        encoded = [char_to_index[char] for char in seed_text] # 현재 시퀀스에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre') # 데이터에 대한 패딩
        encoded = to_categorical(encoded, num_classes=len(char_to_index))

        # 입력한 X(현재 시퀀스)에 대해서 y를 예측하고 y(예측한 문자)를 result에 저장.
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        for char, index in char_to_index.items():
            if index == result:
                break

        # 현재 시퀀스 + 예측 문자를 현재 시퀀스로 변경
        seed_text = seed_text + char

        # 예측 문자를 문장에 저장
        sentence = sentence + char

    # n번의 다음 문자 예측이 끝나면 최종 완성된 문장을 리턴.
    sentence = init_text + sentence
    return sentence
print(sentence_generation(model, char_to_index, 10, 'I get on w', 80))
