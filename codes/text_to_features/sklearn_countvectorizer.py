# Author: Swapnil Gaikwad - NLP practitioner
# Created on : 29th April 2018
# tools used : nltk, sklearn, spacy, re


import sklearn
from sklearn.feature_extraction.text import CountVectorizer 
import spacy
nlp = spacy.load('en')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

class text_to_features():

	def __init__(self):
		self.vectorizer = CountVectorizer(stop_words='english')
		self.lemmatizer = WordNetLemmatizer()

	def getText(self):
		text = "Zebras are several species of African equids (horse family) united by their distinctive black and white stripes. Their stripes come in different patterns, unique to each individual. They are generally social animals that live in small harems to large herds. Unlike their closest relatives, horses and donkeys, zebras have never been truly domesticated. There are three species of zebras: the plains zebra, the Grevy's zebra and the mountain zebra. The plains zebra and the mountain zebra belong to the subgenus Hippotigris, but Grevy's zebra is the sole species of subgenus Dolichohippus. The latter resembles an ass, to which it is closely related, while the former two are more horse-like. All three belong to the genus Equus, along with other living equids. The unique stripes of zebras make them one of the animals most familiar to people. They occur in a variety of habitats, such as grasslands, savannas, woodlands, thorny scrublands, mountains, and coastal hills. However, various anthropogenic factors have had a severe impact on zebra populations, in particular hunting for skins and habitat destruction. Grevy's zebra and the mountain zebra are endangered. While plains zebras are much more plentiful, one subspecies, the quagga, became extinct in the late 19th century - though there is currently a plan, called the Quagga Project, that aims to breed zebras that are phenotypically similar to the quagga in a process called breeding back."
		return text

	def sentTokenize(self, text):
		texts = []
		texts = sent_tokenize(text)
		return texts

	def cleanText(self, doc):

		doc = doc.replace("<br>","")
		doc = doc.replace(":","")
		doc = doc.replace("</br>","")
		doc = doc.replace("<b>","")
		doc = doc.replace("</b>","")
		doc = doc.replace("?","")
		doc = doc.replace("/","")
		doc = doc.replace("\\","")
		doc = doc.replace("-","")
		doc = doc.replace("\/","")
		doc = doc.replace("\n","")
		doc = doc.replace(")","")
		doc = doc.replace("(","")
		doc = doc.replace(".","")
		doc = doc.replace("'","")
		doc = doc.lower().strip()
		return doc

	def cleanSent(self, sent):

		new = []
		sent = self.cleanText(sent)
		sent = nlp(sent)
		for w in sent:
			if (w.is_stop == False) & (w.pos_ != "PUNCT"):
				new.append(w.string.strip())
			c = " ".join(str(x) for x in new)
		return c

	def nltk_lemmatizer(self, sent):
		return " ".join([self.lemmatizer.lemmatize(word) for word in word_tokenize(sent) ])

	def createCountVectors(self, lemm_text):
		features = self.vectorizer.fit_transform(lemm_text)
		print "\n features == ", features
		feature_names = self.vectorizer.get_feature_names()
		print "\n number of words = ",len(feature_names),
		print"\n feature_names ", feature_names 
		return features
		
		""" 
		These created feature vectors from text can be used to train any machine learning classifier, for the applications like : sentiment analysis, detecting whether the email is email is spam or not etc. 
		
		"""

	def prepareData(self):
		text = self.getText()
		texts = self.sentTokenize(text)
		texts = [unicode(sent) for sent in texts]
		clean_text = [self.cleanSent(sent) for sent in texts]
		lemm_text = [self.nltk_lemmatizer(sent) for sent in clean_text]
		print "\n lemm_text ",lemm_text, len(lemm_text)
		features = self.createCountVectors(lemm_text)

if __name__ == '__main__':
	obj = text_to_features()
	obj.prepareData()