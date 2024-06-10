import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from tqdm import tqdm

# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train_data = pd.read_table('ratings.txt')
# print(train_data[:5])
# print(len(train_data))
# print(train_data.isnull().values.any())
train_data = train_data.dropna(how = 'any') #  NULL 값 존재 행 제거
# print(train_data.isnull().values.any())
# print(train_data)
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]","")
# print(train_data[:5])

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()

tokenized_data = []
for sentence in tqdm(train_data['document']):       # tqdm : for문 진행 상황을 알려줌.
    tokenized_sentence = okt.morphs(sentence, stem=True)
    # print(tokenized_sentence)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    tokenized_data.append(stopwords_removed_sentence)

# print('리뷰의 최대 길이 :',max(len(review) for review in tokenized_data))
# print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))       # map : tokenized_data의 요소마다 len 구함
# plt.hist([len(review) for review in tokenized_data], bins=50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()

from gensim.models import Word2Vec

model = Word2Vec(sentences = tokenized_data, vector_size=100, window=5, min_count=5, workers=4, sg=0)
print(model.wv.vectors.shape)
print(model.wv.most_similar("최민식"))
print(model.wv.most_similar("히어로"))