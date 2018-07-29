import nltk
import random
from nltk.corpus import qc
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

#List of Categories :- Int -> 1, Dec -> 0
categoryInterrogative = "Int"
categoryDeclarative = "Dec"

# print(qc.raw("test.txt"))

stop_words = open("Stopwords.txt")
stop_words = stop_words.read()

qc_train = qc.tuples("train.txt")
traindocuments = [x[1] for x in qc_train]

# print(traindocuments)

qc_testInt = qc.tuples("test.txt")
testdocuments = [x[1] for x in qc_testInt]

testDec = open("RawTestingDataDeclarative.txt").read()
testDec = nltk.sent_tokenize(testDec)

# random.shuffle(testdocuments)

# print(testdocuments)

# print(documents)

def findFeaturesOneGram(documents):
	features = {}
	for sentence in documents:
		words = nltk.word_tokenize(sentence)
		tagged = nltk.pos_tag(words)
		firstWord = tagged[0][0]
		firstTag = tagged[0][1]
		tag = firstTag
		if tag == "WDT" or tag == "WP" or tag == "WP$" or tag == "WRB":
			features[firstWord] = True

		lastWord = tagged[-1][0]
		if lastWord == "?":
			features[lastWord] = True

	return features

# findFeatures(documents)
trainfeaturesets = []
trainfeaturesets.append((findFeaturesOneGram(traindocuments), categoryInterrogative))
# print(trainfeaturesets)

testfeaturesets = []
testfeaturesets.append((findFeaturesOneGram(testdocuments), categoryInterrogative))
testfeaturesets.append((findFeaturesOneGram(testDec), categoryDeclarative))
print(testfeaturesets)

classifier = nltk.NaiveBayesClassifier.train(trainfeaturesets)
print("Naive Bayes Classifier Algo Accuracy Percent:", (nltk.classify.accuracy(classifier, testfeaturesets))*100)

