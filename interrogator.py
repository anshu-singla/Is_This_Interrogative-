from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello there Mr. Singh, how are you doing today? The weather is great and Python is pretty good itself. The sky is bluish-red and you should not eat anything not edible!"


print(sent_tokenize(example_text))

print("----------------------------")

print(word_tokenize(example_text))