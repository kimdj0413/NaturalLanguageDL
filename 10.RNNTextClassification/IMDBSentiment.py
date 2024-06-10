import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data()
num_classes = len(set(y_train))

# print('훈련용 리뷰 개수 : {}'.format(len(X_train)))
# print('테스트용 리뷰 개수 : {}'.format(len(X_test)))
# print('카테고리 : {}'.format(num_classes))

reviews_length = [len(review) for review in X_train]
# print('리뷰의 최대 길이 : {}'.format(np.max(reviews_length)))
# print('리뷰의 평균 길이 : {}'.format(np.mean(reviews_length)))

unique_elements, counts_elements = np.unique(y_train, return_counts=True)
# print("각 레이블에 대한 빈도수:")
# print(np.asarray((unique_elements, counts_elements)))

word_to_index = imdb.get_word_index()
index_to_word = {}
for key, value in 