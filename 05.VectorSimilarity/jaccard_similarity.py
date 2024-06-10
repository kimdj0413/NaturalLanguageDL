doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

# 토큰화
tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()

# print('문서1 :',tokenized_doc1)
# print('문서2 :',tokenized_doc2)
union = set(tokenized_doc1).union(set(tokenized_doc2))
# print('문서1과 문서2의 합집합 :',union)
intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
# print('문서1과 문서2의 교집합 :',intersection)
print('자카드 유사도 :',len(intersection)/len(union))