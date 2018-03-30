"""
Author : Swapnil Gaikwad
tolls : nltk, sklearn, numpy, re (regular expressions), Beautifulsoup
Title: Document classification in 2 classes using naive bayes algorithm (binary classifier) => 0 - personal query, 1 - question query 
features used : tfidf
"""

from nltk.corpus import stopwords
import numpy as np
import re
from bs4 import Beautifulsoup
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class NaiveBayes():

	def __init__(self):
		self.grid_flag = 0
		self.vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf = True, max_df = 0.5)
		self.stopwords = stopwords.words('english')
		self.labels = ['personal','question']
		self.model = self.train()

	def cleanText(self, doc):
		doc = doc.lower().strip()
		soup = Beautifulsoup(doc, 'html.parser')
		doc = str(soup.get_text())
		doc = doc.replace(":","")
		doc = doc.replace("?","")
		doc = doc.replace("/","")
		doc = doc.replace("\\","")
		doc = doc.replace("-","")
		doc = doc.replace("\/","")
		doc = doc.replace("\n","")
		doc = doc.replace(")","")
		doc = doc.replace("(","")
		doc = doc.replace(".","")
		doc = doc.encode('ascii','ignore')
		return doc

	def getdata(self,):

		personal = []
		question = []

		with open('./question.txt') as fq:
			question = fq.readlines()
			question = [lines.replace("\n", "").lower() for lines in question]
			question = [ self.cleanText(sent) for sent in question]
		with open('./personal.txt') as fp:
			personal = fp.readlines()
			personal = [lines.replace("\n", "").lower() for lines in personal]
			personal = [ self.cleanText(sent) for sent in personal]

		print "\n len(personal) = ",len(personal)
		print "\n len(question) = ",len(question)

		x = personal + question
		y = [0] * len(personal) + [1] * len(question)
		y = np.array(y).reshape(len(y),1)
		print "\n total data size = ",len(x)
		x_train, x_text, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=42)
		print "\n data size after cross validation = ",len(x_train)
		return x_train, x_text, y_train, y_test

	def train(self,):
		x_train, x_test, y_train, y_test = self.getdata()
		train_vector = self.vectorizer.fit_transform(x_train)
		test_vector = self.vectorizer.transform(x_test)
		
		self.clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
		self.model = self.clf.fit(train_vector, y_train.ravel())
		print "\n training done ::"

		pred = self.model.predict(test_vector)
		print "\ntest accuracy ==>> ", accuracy_score(y_test, pred)

		print "\n classification_report ::\n",classification_report(y_test, pred)
		return self.model

	def test(self, query):
		print "\n query ==>> ",query
		query_vector = self.vectorizer.transform([query])
		pred_class = self.model.predict(query_vector)
		print "\n predicted class ==>> ",self.labels[pred_class[0]]



if __name__ == '__main__':
		nb = NaiveBayes()
		nb.test("what is my current role")