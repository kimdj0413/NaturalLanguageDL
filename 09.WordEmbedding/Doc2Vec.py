import pandas as pd
# from konlpy.tag import Mecab
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
import pickle

## mecab은 구글 코랩에서 사용 후 pickle로 리스트 로컬 저장.

df = pd.read_csv('./data/dart.csv', sep=',')
df = df.dropna()
# print(df)

# mecab = Mecab()

tagged_corpus_list = []

# for index, row in tqdm(df.iterrows(), total=len(df)):
#   text = row['business']
#   tag = row['name']
#   tagged_corpus_list.append(TaggedDocument(tags=[tag], words=mecab.morphs(text)))

# with open('list.pkl', 'wb') as file:
#     pickle.dump(tagged_corpus_list, file)

# print('문서의 수 :', len(tagged_corpus_list))
# tagged_corpus_list[0]

with open('./data/list.pkl', 'rb') as file:
    tagged_corpus_list = pickle.load(file)

# print(tagged_corpus_list[0])

from gensim.models import doc2vec

# model = doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, workers=8, window=8)
# model.build_vocab(tagged_corpus_list)
# model.train(tagged_corpus_list, total_examples=model.corpus_count, epochs=20)
# model.save('./model/dart.doc2vec')

model = doc2vec.Doc2Vec.load('./model/dart.doc2vec')
similar_doc = model.dv.most_similar('동화약품')
print(similar_doc)
