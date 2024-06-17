import sentencepiece as spm
import pandas as pd
import urllib.request
import csv

# urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="./data/IMDb_Reviews.csv")

train_df = pd.read_csv('./data/IMDb_Reviews.csv')
# print(train_df)

# with open('./data/imdb_review.txt', 'w', encoding='utf8') as f:
    # f.write('\n'.join(train_df['review']))

# spm.SentencePieceTrainer.Train('--input=./data/imdb_review.txt --model_prefix=./model/imdb --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')

vocab_list = pd.read_csv('./model/imdb.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
# print(vocab_list.sample(10))
# print(len(vocab_list))

sp = spm.SentencePieceProcessor()
vocab_file = "./model/imdb.model"
print(sp.load(vocab_file))

lines = ["I didn't at all think of it this way.",
         "I have waited a long time for someone to film"]

# for line in lines:
#     print(line)
#     print(sp.encode_as_pieces(line))
#     print(sp.encode_as_ids(line))
#     print()

print(sp.GetPieceSize)
print(sp.IdToPiece(430))
print(sp.PieceToId('_character'))
print(sp.DecodeIds([41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91]))
