import nltk
import random
from nltk.corpus import qc
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
                    "does the car break down often","can you drive me home","where did you find me", "Are you okay with this", "Will they ever leave",
                    "can it fly from here to target","could you find it for me, Are you coming with me or not"]
Sample_Answer = ["The asnwer is so simple", "This looks like an easy thing to do.", "This must be doable.", "You know you should do what he is telling you to.", "This is an orange", "That was a train.", "Where there is a will, there is a way"]               

wh_words_tags = ["WDT", "WP", "WP$", "WRB"]

qc_train = qc.tuples("train.txt")
traindocuments = [x[1] for x in qc_train]
traindocuments = traindocuments[:250]

trainDec = open("RawTestingDataDeclarative.txt").read()
trainDec = nltk.sent_tokenize(trainDec)

# print(traindocuments)

qc_testInt = qc.tuples("test.txt")
testdocuments = [x[1] for x in qc_testInt]
testdocuments = testdocuments + Sample_Questions

testDec = Sample_Answer

# random.shuffle(testdocuments)

# print(testdocuments)

# print(documents)

def ifTagExistsOnFirst(featureOfAParticularSentence, firstTag, tag):
	if firstTag == tag:
		featureOfAParticularSentence[tag] = True
	else:
		featureOfAParticularSentence[tag] = False

def findFeaturesOneGram(documents, categoryDefined):
	features = []
	for sentence in documents:
		featureOfAParticularSentence = {}
		words = nltk.word_tokenize(sentence)
		tagged = nltk.pos_tag(words)
		firstWord = tagged[0][0]
		firstTag = tagged[0][1]

		# 1st Feature
		ifTagExistsOnFirst(featureOfAParticularSentence, firstTag, "WDT")

		# 2nd Feature
		ifTagExistsOnFirst(featureOfAParticularSentence, firstTag, "WP")

		# 3rd Feature
		ifTagExistsOnFirst(featureOfAParticularSentence, firstTag, "WP$")

		# 4th Feature
		ifTagExistsOnFirst(featureOfAParticularSentence, firstTag, "WRB")

		# 5th Feature
		lastWord = tagged[-1][0]
		lastTag = tagged[-1][1]
		questionSign = "?"
		if lastWord == questionSign:
			featureOfAParticularSentence[questionSign] = True
		else:
			featureOfAParticularSentence[questionSign] = False
		

		myDict = (featureOfAParticularSentence, categoryDefined)
		features.append(myDict)
	return features

# findFeatures(documents)
trainfeaturesets = findFeaturesOneGram(traindocuments, categoryInterrogative) + findFeaturesOneGram(trainDec, categoryDeclarative)
random.shuffle(trainfeaturesets)
# print(trainfeaturesets)

testfeaturesets = findFeaturesOneGram(testdocuments, categoryInterrogative) + findFeaturesOneGram(testDec, categoryDeclarative)
random.shuffle(testfeaturesets)
# print(testfeaturesets)

classifier = nltk.NaiveBayesClassifier.train(trainfeaturesets)
print("Naive Bayes Classifier Algo Accuracy Percent:", (nltk.classify.accuracy(classifier, testfeaturesets))*100)


