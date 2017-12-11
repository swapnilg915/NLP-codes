# 1. lemmatizer and Stemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from stemmer.porter2 import stem
import spacy

#### spacy object
spacy_obj = spacy.load('en')

#### stemmers
nltk_porter_stemmer = PorterStemmer()
porter_stem = stem()

#### lemmatizers
nltk_lem = WordNetLemmatizer()


sentences = ["this is a lovely place","that is a playable delivery","this is amazing life!","I am learning Data Science","My knowledge is getting multiplied"]

for sent in sentences:
	print "\n original = ",sent
	print "\n stemming operations :: \n"
	print "\n stemming = ",nltk_porter_stemmer.stem(sent),"\n"

	print "\n lemmatization operations :: "
	print "\n lemmatized = ",nltk_lem.lemmatize(sent),"\n"
	for tok in spacy_obj(sent):
		print tok, tok.lemma_
