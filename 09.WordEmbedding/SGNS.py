import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer

# dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
# documents = dataset.data

# news_df = pd.DataFrame({'document':documents})

# news_df.to_csv('C:/personal_coding/NaturalLanguage/data/news_df.csv', index=False)

news_df = pd.read_csv('C:/NaturalLanguage/data/news_df.csv')
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-z]"," ")
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x:' '.join([w for w in x.split() if len(w)>3]) if isinstance(x, str) else x)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower() if isinstance(x, str) else x)
news_df.replace("",float("NaN"), inplace=True)
news_df.dropna(inplace=True)
# print(news_df.isnull().values.any())
# print(len(news_df))

stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = tokenized_doc.to_list()

drop_train = [index for index, sentence in enumerate(tokenized_doc) if len(sentence) <= 1]
tokenized_doc = np.delete(tokenized_doc, drop_train, axis=0)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokenized_doc)

word2idx = tokenizer.word_index
idx2word = {value : key for key, value in word2idx.items()}
encoded = tokenizer.texts_to_sequences(tokenized_doc)
vocab_size = len(word2idx) + 1 
print('단어 집합의 크기 :', vocab_size)

from tensorflow.keras.preprocessing.sequence import skipgrams
skip_grams = [skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:10]]
