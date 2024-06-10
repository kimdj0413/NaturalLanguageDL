# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import SimpleRNN

# model = Sequential()
# model.add(SimpleRNN(3, batch_input_shape=(8,2,10),return_sequences=True))
# print(model.summary())

import numpy as np

timesteps = 10      # 시점의수(보통 문장의 길이)
input_dim = 4       # 입력차원(단어 벡터의 차원)
hidden_units = 8    # 은닉 상태의 크기(메모리 셀 용량)

inputs = np.random.random((timesteps, input_dim))
hidden_state_t = np.zeros((hidden_units))