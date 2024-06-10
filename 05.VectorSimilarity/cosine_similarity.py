import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('movies_metadata.csv', low_memory=False)
data = data.head(20000)
# print(data.head(2))
# print(data['overview'].isnull().sum())
data['overview'] = data['overview'].fillna('')
# print(data['overview'].isnull().sum())
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
# print(tfidf_matrix.shape)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# print(cosine_sim.shape)
title_to_index = dict(zip(data['title'], data.index))
# idx = title_to_index['Father of the Bride Part II']
# print(idx)
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = title_to_index[title]
    # enumerate -> 인덱스와 요소를 동시에 처리
    sim_scores = list(enumerate(cosine_sim[idx]))
    # x[1], 즉 요소 값을 비교하여 sorted
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [idx[0] for idx in sim_scores]
    return data['title'].iloc[movie_indices]

print(get_recommendations('The Dark Knight Rises'))