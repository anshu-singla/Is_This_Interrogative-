import nltk
import random
from nltk.corpus import qc
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# print(qc.raw("test.txt"))

qc_train = qc.tuples("test.txt")
documents = [x[1] for x in qc_train]

print(documents)

for i in documents:
	tokenized_woords = word_tokenize(i)
	print(tokenized_woords)