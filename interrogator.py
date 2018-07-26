import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer

example_text = "Hello there Mr. Singh, how are you doing today? The weather is great and Python is pretty good itself. The sky is bluish-red and you should not eat anything not edible!"
train_text = gutenberg.raw('austen-emma.txt')
# example_text = gutenberg.raw("austen-persuasion.txt")



#print(sent_tokenize(example_text))
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


print("----------------------------Custom Tokenizing and Part-of-speech Tagging")
custom_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_tokenizer.tokenize(example_text)

def process_content():
	try:
		for i in tokenized:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			print(tagged)
	except Exception as e:
		print(str(e))

process_content()		