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

Sample_Questions = ["what is the weather like","where are we today","why did you do that","where is the dog","when are we going to leave","why do you hate me","what is the Answer to question 8",
                    "what is a dinosour","what do i do in an hour","why do we have to leave at 6.00", "When is the apointment","where did you go","why did you do that","how did he win","why won’t you help me",
                    "when did he find you","how do you get it","who does all the shipping","where do you buy stuff","why don’t you just find it in the target","why don't you buy stuff at target","where did you say it was",
                    "when did he grab the phone","what happened at seven am","did you take my phone","do you like me","do you know what happened yesterday","did it break when it dropped","does it hurt everyday",
                    "does the car break down often","can you drive me home","where did you find me"
                    "can it fly from here to target","could you find it for me, Are you coming with me or not"]
Sample_Answer = ["The asnwer is so simple", "This looks like an easy thing to do.", "This must be doable.", "You know you should do what he is telling you to.", "This is an orange", "That was a train.", "Where there is a will, there is a way"]               

wh_words_tags = ["WDT", "WP", "WP$", "WRB"]

stop_words = open("Stopwords.txt")
stop_words = stop_words.read()

qc_train = qc.tuples("train.txt")
traindocuments = [x[1] for x in qc_train]
traindocuments = traindocuments[:250]

# print(traindocuments)

qc_testInt = qc.tuples("test.txt")
testdocuments = [x[1] for x in qc_testInt]
testdocuments = testdocuments + Sample_Questions

trainDec = open("RawTestingDataDeclarative.txt").read()
trainDec = nltk.sent_tokenize(trainDec)

testDec = Sample_Questions

# random.shuffle(testdocuments)

# print(testdocuments)

# print(documents)

def findFeaturesOneGram(documents, isInterrogative):
	features = {}
	for sentence in documents:
		words = nltk.word_tokenize(sentence)
		tagged = nltk.pos_tag(words)
		firstWord = tagged[0][0]
		firstTag = tagged[0][1]
		tag = firstTag
		if tag == "WDT" or tag == "WP" or tag == "WP$" or tag == "WRB":
			features[tag] = isInterrogative
		else:
			features[tag] = not isInterrogative	

		lastWord = tagged[-1][0]
		lastTag = tagged[-1][1]
		if lastWord == "?":
			features[lastWord] = isInterrogative
		else:
			features[lastWord] = not isInterrogative

	return features

# findFeatures(documents)
trainfeaturesets = []
trainfeaturesets.append((findFeaturesOneGram(traindocuments, True), categoryInterrogative))
trainfeaturesets.append((findFeaturesOneGram(trainDec, False), categoryDeclarative))
# print(trainfeaturesets)

testfeaturesets = []
testfeaturesets.append((findFeaturesOneGram(testdocuments, True), categoryInterrogative))
testfeaturesets.append((findFeaturesOneGram(testDec, False), categoryDeclarative))
# print(testfeaturesets)

classifier = nltk.NaiveBayesClassifier.train(trainfeaturesets)
print("Naive Bayes Classifier Algo Accuracy Percent:", (nltk.classify.accuracy(classifier, testfeaturesets))*100)

# def findFeaturesBiGram(documents, isInterrogative, n=2):
# 	features = {}
# 	for sentence in documents:
# 		words = nltk.word_tokenize(sentence)
# 		tagged = nltk.pos_tag(words)
# 		ngram_vocab = ngrams(tagged, n)
# 		for ng in ngram_vocab:
# 			ngk
# 		if tag == "WDT" or tag == "WP" or tag == "WP$" or tag == "WRB":
# 	return features

