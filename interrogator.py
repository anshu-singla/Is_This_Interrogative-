import nltk
import Stopwords
from nltk.tokenize import word_tokenize
# from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer

example_text = "Is there any way to do this? Hello there Mr. Singh, how are you doing today? The weather is great and Python is pretty good itself. The sky is bluish-red and you should not eat anything not edible! But who are you? and why are you saying this is not possible."
train_text = gutenberg.raw('austen-emma.txt')
# example_text = gutenberg.raw("austen-persuasion.txt")



# print(sent_tokenize(example_text))
# print("----------------------------Tokenizing")
# words = word_tokenize(example_text)
# print(words)



# print("----------------------------Stopwords")
# stop_words = set(stopwords.words("english"))
# print(stop_words)



# print("----------------------------Stemming")
# ps = PorterStemmer()
# stemmed_words = [ps.stem(w) for w in words ]
# print(stemmed_words)


print("----------------------------Custom Tokenizing, Part-of-speech Tagging, Chunking-----")
custom_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_tokenizer.tokenize(example_text)

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			print(tagged)

			# chunkGram = r"""Chunk: {<W.+>*} """

			# chunkParser = nltk.RegexpParser(chunkGram)
			# chunked = chunkParser.parse(tagged)
			# print(chunked)
			# chunked.draw()

	except Exception as e:
		print(str(e))

process_content()		


print("----------------------------Lemmatizing")
# Same as stemming but the end result is a meaningfull word
# DOES NOT WORK ON CAPITAL LETTER BEGINING WORDS
# Defualt paramter for lemmatizing in POStag is a noun , "n"..So if you are trying to pass anything else to lemmatize, try and use specific post tag for that.

# lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize("cats"))
# print(lemmatizer.lemmatize("better"))
# print(lemmatizer.lemmatize("better", pos="a"))
# print(lemmatizer.lemmatize("best", pos="a"))
# print(lemmatizer.lemmatize("ride"))
# print(lemmatizer.lemmatize("rode", pos = "v"))
# print(lemmatizer.lemmatize("ridden", pos = "v"))

