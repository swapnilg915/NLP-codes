#!/usr/bin/python
# -*- coding: utf-8 -*-

import gearman
import os
import glob
import traceback
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from trainingDataCreator import DataCreator

model_path = 'Models/svc_model/'

class PickleOperation(object):
   
    def __init__(self, ):        
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    def readPickle(self, filename):
        classifier = joblib.load(model_path + filename + '.pkl')
        return classifier

    def writePickle(self, clf, filename):
        joblib.dump(clf,model_path + filename +'.pkl')

class ExperinceClassifier(PickleOperation):
    
    def __init__(self,):
        
        PickleOperation.__init__(self)
        # self.gm_worker = gearman.GearmanWorker(["localhost:4730"])
        # self.gm_worker.register_task("experience_classifier", self.testQuery)
        self.data_obj = DataCreator()
        self.svm_model = "experience_clf"
        self.vectorizer = TfidfVectorizer( sublinear_tf=True, max_df = 0.5, use_idf=False)
        self.targets = ["other","experience"]

    def getData(self,):
        
        features = []
        labels = []
        features, labels = self.data_obj.CreateData()
        features_train,features_test,labels_train,labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)
        return features_train,labels_train,features_test,labels_test

    
    def train(self,):
        
        try:
            features_train,labels_train,features_test,labels_test = self.getData()
            # print "\nlen(features_train) = ",len(features_train)
            # print "\nlen(labels_train) = ",len(labels_train)
            # print "\nlen(features_test) = ",len(features_test)
            # print "\nlen(labels_test) = ",len(labels_test)
            features_train_vector = self.vectorizer.fit_transform(features_train)
            features_test_vector = self.vectorizer.transform(features_test)
            
            ## clf1 => LinearSVC using GridSearchCV
            # svm = LinearSVC(random_state = 42)
            # parameters = [{
            #     'C':[0.5,1,4,10,15,20,25,30,35,40,50,100,500,1000]
            # }]
            # self.clf = GridSearchCV(svm,parameters)
            
            ## clf2 => simple LinearSVC
            # self.clf = LinearSVC(random_state = 42)
            
            ##clf3 => SVC using GridSearchCV
            # svm = SVC(random_state = 42)
            # parameters = [{
            #     'C':[0.5,1,4,10,15,20,25,30,35,40,50,100,500,1000],
            #     'gamma':[1e-4,1e-3,1e-2,1e-1,0.5,1,1e1,1e2,1e3,1e4],
            #     'kernel':['rbf']
            # }]
            # self.clf = GridSearchCV(svm,parameters)
            
            ## clf4 best of clf3
            self.clf = SVC(kernel='rbf', C=20, gamma=0.1, random_state = 42)
            
            self.clf.fit(features_train_vector, labels_train)
            
            # print "\nBest parameters set found on development set: ",self.clf.best_params_
            # print "\n Best estimator found = ",self.clf.best_estimator_
            
            self.writePickle(self.clf, self.svm_model)
            preds = self.clf.predict(features_test_vector)
        
        except Exception as e:
            print "\n Error in train : ",e
            print traceback.format_exc()
        
        # print "\n clasifier score = ",self.clf.score(features_train_vector, labels_train)
        # print "\n test acc = ",accuracy_score(preds, labels_test)
        # print "\n classification report = \n ",classification_report(labels_test,preds)
        # print "\n confusion_matrix = \n ",confusion_matrix(labels_test, preds)
        # print "=="*50,"\nResults : \n","=="*50
        # for (sample, pred) in zip(features_test, preds):
        #     print "\nline : ",sample, " = ", pred
        
        
    def testQuery(self, query):
        
        cleaned_query = self.data_obj.cleanText(query)
        self.train()
        # self.clf = self.readPickle(self.svm_model)
        query_vector = self.vectorizer.transform([cleaned_query])
        pred = self.clf.predict(query_vector)
        if len(pred):
            pred_class = self.targets[pred[0]]
        classification_result = {"query":cleaned_query,"class":pred_class}
        print "\n classification_result = ",classification_result
        return classification_result
        
    
if __name__ == '__main__':
    obj = ExperinceClassifier()
    # obj.train()
    obj.testQuery("Apollo Technical Education Foundation (Initiative by  Apollo Tyres Ltd),Vadodara Oct 2012-Dec 2014")
