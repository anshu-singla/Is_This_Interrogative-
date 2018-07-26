from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

example_text = "Hello there Mr. Singh, how are you doing today? The weather is great and Python is pretty good itself. The sky is bluish-red and you should not eat anything not edible!"


#print(sent_tokenize(example_text))

print("----------------------------Tokenizing")

words = word_tokenize(example_text)
print(words)

print("----------------------------Stopwords")

stop_words = set(stopwords.words("english"))
print(stop_words)
