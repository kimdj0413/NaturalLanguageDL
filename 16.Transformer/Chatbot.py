import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow as tf
import tensorflow_datasets as tfds

# urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="./data/ChatBotData.csv")
train_data = pd.read_csv('./data/ChatBotData.csv')
# print(train_data.head())
# print(len(train_data))
print(train_data.isnull().sum())

##      구두점 사이에 공백 집어넣기
questions = []
for sentence in train_data['Q']:
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['A']:
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    sentence.strip()
    answers.append(sentence)

# print(questions[:5])
# print(answers[:5])

##      서브워드텍스트 인코더
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size = 2**13
)