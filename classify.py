import nltk
import random
from nltk.corpus import qc
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# print(qc.raw("test.txt"))

stop_words = open("Stopwords.txt")
stop_words = stop_words.read()

qc_train = qc.tuples("test.txt")
documents = [x[1] for x in qc_train]

# print(documents)

all_words = []
for i in documents:
	for word in word_tokenize(i):
		all_words.append(word)
# tokenized_words = word_tokenize(i)
# all_words.append(word_tokenize(i).words())
# print(tokenized_words)

# Frequency of words

all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))

print(stop_words)
