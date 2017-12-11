
"""
Author : Swapnil Gaikwad
Title : Answer classification in 2 classes using naive bayes, linear svm and logistic regression (binary classifier) => 0 - bad, 1 - good
tools : nltk, sklearn, numpy, re, BeautifulSoup
features used => tfidf
Note : it is a attempt to solve the hackerrank "statistics and machine learning " (hard) challenge. This classifier has acheived upto 78 % of accuracy.
"""

import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

class quoraAnswerClassifier(object):

	def __init__(self,):
		self.nb_classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
		self.linearsvc_classifier = LinearSVC()
		self.Logistic_classifier = LogisticRegression(solver='liblinear', max_iter=11, multi_class='ovr', n_jobs=-1,random_state = 42)
		

	def getData(self, filename):

	    # input_file = open(filename)
	    # traindata = input_file.readlines()
	    traindata = 

	    features = []
	    targets = []
	    for line in traindata:
	        formatted_line = line.strip("\n")
	        target_i = formatted_line.split(" ")[1]
	        feature_i = re.sub(r"(\d+):", "", formatted_line).split(" ")[2:]

	        targets.append(target_i)
	        features.append(feature_i)

		matrix_features = np.array(features).astype(np.float)
		vector_targets = np.array(targets).astype(np.int)
	    input_file.close()

	    return matrix_features, vector_targets


	def normalize_data(self, matrix_features):
	    max_features = matrix_features.max(axis = 0)
	    max_features = (max_features + (max_features == 0))
	    return matrix_features / max_features


	def trainModel(self, all_features, all_targets, test_features, test_targets, test_name):

		self.nb_classifier.fit(all_features, all_targets)
		print self.nb_classifier.predict(test_features)
		accuracy = self.nb_classifier.score(test_features, test_targets)
		print "\n naive bayes accuracy == ",accuracy

		self.linearsvc_classifier.fit(all_features, all_targets)
		print self.linearsvc_classifier.predict(test_features)
		accuracy = self.linearsvc_classifier.score(test_features, test_targets)
		print "\n linear svc accuracy == ",accuracy

		self.Logistic_classifier.fit(all_features, all_targets)
		print self.Logistic_classifier.predict(test_features)
		accuracy = self.Logistic_classifier.score(test_features, test_targets)
		print "\n logistic regression accuracy == ",accuracy


	def test_raw_dataset(self):
		all_features, all_targets = self.getData()
		test_features, test_targets = self.getData()
		all_features = self.normalize_data(all_features)
		test_features = self.normalize_data(test_features)
		self.trainModel(all_features, all_targets, test_features, test_targets, data_flag)


if __name__ == '__main__':
	train_dataset_filename = 'dataset/hackerrank_train.txt'
	test_filename = 'dataset/hackerrank_test.txt'
	obj = quoraAnswerClassifier()
	obj.test_raw_dataset()