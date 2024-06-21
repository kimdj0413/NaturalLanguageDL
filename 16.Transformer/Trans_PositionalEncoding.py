import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

##      포지셔널 인코딩
class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, position, d_model):    # position : 최대 시퀸스 길이, d_model : 임베딩 벡터의 차원
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):   # position과 임베딩 차원 인덱스 i에 대한 각도를 구함
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    # tf.pow : 거듭제곱 연산, tf.cast : 데이터 타입 변환
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)

    # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
    sines = tf.math.sin(angle_rads[:, 0::2])

    # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
    cosines = tf.math.cos(angle_rads[:, 1::2])

    angle_rads = np.zeros(angle_rads.shape)
    angle_rads[:, 0::2] = sines
    angle_rads[:, 1::2] = cosines
    pos_encoding = tf.constant(angle_rads)
    pos_encoding = pos_encoding[tf.newaxis, ...]

    print(pos_encoding.shape)
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
# 파라미터 설정
max_length = 50
d_model = 128

# 레이어와 입력 생성
sample_pos_encoding = PositionalEncoding(max_length, d_model)
inputs = tf.random.uniform((2, 40, d_model))  # (batch_size, sequence_length, d_model)

# 레이어 호출
outputs = sample_pos_encoding(inputs)

# print(outputs.shape)  # (2, 40, 512)

plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 128))
plt.ylabel('Position')
plt.colorbar()
plt.show()