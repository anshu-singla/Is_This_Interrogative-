import nltk
import random
from nltk.corpus import qc
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

#List of Categories :- Int -> 1, Dec -> 0
categoryInterrogative = "Int"

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
testdocuments = testdocuments + testDec
random.shuffle(testdocuments)

# print(testdocuments)

# print(documents)

def findFeaturesOneGram(documents):
	features = {}
	for sentence in documents:
		words = nltk.word_tokenize(sentence)
		tagged = nltk.pos_tag(words)
		# wordPosTagDict = {}
		for i in range(0, len(tagged)):
			word = tagged[i][0]
			word = word.lower()
			tag = tagged[i][1]
			if tag == "WDT" or tag == "WP" or tag == "WP$" or tag == "WRB":
				# wh_pos = tag
				# wh_word = word
				features[word] = True
				# next_word = tagged[i+1][0]
				# next_tag = tagged[i+1][1]
				# wh_bi_gram.append(wh_word)
			elif word == "?":
				features[word] = True
			# else :
			# 	features[word] = False

		# features.append((wordPosTagDict, categoryInterrogative))
	return features

# findFeatures(documents)
trainfeaturesets = []
# trainfeaturesets.append((findFeaturesOneGram(traindocuments), categoryInterrogative))
# print(trainfeaturesets)

testfeaturesets = []
# testfeaturesets.append((findFeaturesOneGram(testdocuments), categoryInterrogative))
# print(testfeaturesets)

# classifier = nltk.NaiveBayesClassifier.train(trainfeaturesets)
# print("Naive Bayes Classifier Algo Accuracy Percent:", (nltk.classify.accuracy(classifier, testfeaturesets))*100)
# classifier.show_most_informative_features(15)

