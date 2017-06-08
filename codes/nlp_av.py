# 1. lemmatizer and Stemmer

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

sentences = ["this is a lovely place","that is a playable delivery","this is amazing life!","I am learning Data Science","My knowledge is getting multiplied"]
for sent in sentences:
	print "\n original = ",sent
	print "\n lemmatized = ",lem.lemmatize(sent),"\n"
	print "\n stemming = ",stemmer.stem(sent),"\n"

