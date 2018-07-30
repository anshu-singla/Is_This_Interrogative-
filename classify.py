import nltk
import random
import re
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
                    "can it fly from here to target","could you find it for me", "Are you coming with me or not", "Shall i pick you up tomorrow night?"]
Sample_Answer = ["The answer is so simple", "This looks like an easy thing to do.", "This must be doable.", "You know you should do what he is telling you to.", "This is an orange", "That was a train.", "Where there is a will, there is a way"]               

WH_WORDS = ["WDT", "WP", "WP$", "WRB"]
VERB_TAGS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

qc_train = qc.tuples("train.txt")
traindocuments = [x[1] for x in qc_train]
# traindocuments = traindocuments[:500]

trainDec = open("RawTestingDataDeclarative.txt").read()
trainDec = nltk.sent_tokenize(trainDec)

# print(traindocuments)

qc_testInt = qc.tuples("test.txt")
testdocuments = [x[1] for x in qc_testInt]
testdocuments = testdocuments + Sample_Questions
random.shuffle(testdocuments)

testDec = Sample_Answer

def isSubjectVerbInversionPresent(taggedList):
	verb = [taggedList[0][1] in VERB_TAGS]
	# subject = [re.search("<NN.+>*", taggedList[1][1])]
	if verb:
		return True
	else:
		return False

def ifWHTagExistsOnFirstThenVerbOnSecond(taggedList):
	whWord = [taggedList[0][1] in WH_WORDS]
	verb = [taggedList[1][1] in VERB_TAGS]
	if verb and whWord:
		return True
	else:
		return False

def findFeaturesOneGram(documents, categoryDefined):
	features = []
	for sentence in documents:
		# print(sentence)
		featureOfAParticularSentence = {}
		words = nltk.word_tokenize(sentence)
		tagged = nltk.pos_tag(words)
		tags = [x[1] for x in tagged]
		# print(tags)
		firstWord = tagged[0][0]
		firstTag = tagged[0][1]

		# First feature includes the words that are starting with WH_words and next word is a Verb
		featureNameWHWord = "WHV"
		featureOfAParticularSentence[featureNameWHWord] = ifWHTagExistsOnFirstThenVerbOnSecond(tagged)

		# This Feature includes sentences starting with ("Will", "Would", "Can", "Could") impposing a sense of question
		tag = "MD" 
		if firstTag == tag:
			featureOfAParticularSentence[tag] = True
		else:
			featureOfAParticularSentence[tag] = False

		# This feature indicates the question mark at the end of the sentence
		lastWord = tagged[-1][0]
		lastTag = tagged[-1][1]
		questionSign = "?"
		if lastWord == questionSign:
			featureOfAParticularSentence[questionSign] = True
		else:
			featureOfAParticularSentence[questionSign] = False
		
		# Feature : Is subject verb inversion present at starting two indexes
		featureNameSVI = "SVI"
		featureOfAParticularSentence[featureNameSVI] = isSubjectVerbInversionPresent(tagged)

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


