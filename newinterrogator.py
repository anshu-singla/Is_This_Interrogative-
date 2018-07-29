import nltk
import random
from nltk import ngrams
from nltk.corpus import qc
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

#List of Categories :- Int -> 1, Dec -> 0
categoryInterrogative = "Int"
categoryDeclarative = "Dec"

# print(qc.raw("test.txt"))

trainData = open("newTrainData30k.txt").read()
trainData = nltk.sent_tokenize(trainData)

qc_train = qc.tuples("train.txt")
traindocuments = [x[1] for x in qc_train]
trainData = trainData[:10000]

qc_testInt = qc.tuples("test.txt")
testdocuments = [x[1] for x in qc_testInt]

testDec = open("RawTestingDataDeclarative.txt").read()
testDec = nltk.sent_tokenize(testDec)

def findFeatures(documents, isInterrogative):
	features = {}
	for sentence in documents:
		words = nltk.word_tokenize(sentence)
		tagged = nltk.pos_tag(words)
		tagSet = set()
		for x in tagged:
			tagSet.add(x[1])
		ngrams_vocab = ngrams(tagSet, len(tagSet))
		my_dict = dict([(ng, isInterrogative) for ng in ngrams_vocab])
		features = {**features, **my_dict}
	return features

trainfeaturesets = []
trainfeaturesets.append((findFeatures(trainData, True), categoryInterrogative))

testfeaturesets = []
testfeaturesets.append((findFeatures(testdocuments, True), categoryInterrogative))
testfeaturesets.append((findFeatures(testDec, True), categoryDeclarative))

classifier = nltk.NaiveBayesClassifier.train(trainfeaturesets)
print("Naive Bayes Classifier Algo Accuracy Percent:", (nltk.classify.accuracy(classifier, testfeaturesets))*100)
