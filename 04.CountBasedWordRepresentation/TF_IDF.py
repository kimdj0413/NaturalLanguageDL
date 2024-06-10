import pandas as pd
from math import log

docs = [
    '먹고 싶은 사과',
    '먹고 싶은 바나나',
    '길고 노란 바나나 바나나',
    '저는 과일이 좋아요'
]
vocab = list(set(w for doc in docs for w in doc.split()))
# 제너레이터 표현식(메모리 절약된 리스트 컴프리헨션)
# set은 중복 제거
# list는 순서가 있는 집합

# 리스트 컴프리헨션(제너레이터와 비슷)
# words = ['apple', 'banana', 'cherry', 'date']
# long_words = [word for word in words if len(word) >= 5]
# print(long_words)
# ['apple', 'banana', 'cherry']

vocab.sort()
# print(vocab)

N = len(docs)

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df+1))

def tfidf(t, d):
    return tf(t,d)*idf(t)

result = []

#DTM
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t,d))

tf_ = pd.DataFrame(result, columns=vocab)

# print(tf_)

result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))
idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
print(idf_)

result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t,d))
tfidf_ = pd.DataFrame(result, columns = vocab)
print(tfidf_)