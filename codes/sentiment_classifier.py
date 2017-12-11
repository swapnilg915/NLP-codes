"""
Author : Swapnil Gaikwad
Title : sentiment classification in 2 classes (binary classifier) => 0 - negative, 1 - positive
tools : nltk, sklearn, numpy, re, BeautifulSoup
dataset : Amazon customer reviews dataset (size = 1000)
features used => tfidf
"""

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
from bs4 import BeautifulSoup
import numpy as np
from nltk.corpus import stopwords

class SentimentAnalyzer(object):
    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.remove_stopwords = 0
        self.labels = ['negative','positive']
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, max_df=0.5)
        self.clf = LinearSVC()

    
    def cleanText(self,text):
        soup = BeautifulSoup(text.lower().strip(), 'html.parser')
        return str(soup.get_text()).translate(None, string.punctuation).encode('utf-8')
    
    def cleanQuery(self, query):
        query = query.lower().strip()
        query = re.sub('<[^<]+?>', ' ', query)
        clean_query = re.sub(r"<\/?ol>|<\/?li>"," ", query)
        clean_query = re.sub(r'[&]', ' and ', clean_query)
        clean_query = re.sub(r'[|]', ' or ', clean_query)
        clean_query = re.sub(r'[^\w]', ' ', clean_query)
        clean_query = re.sub(r'\s+', ' ', clean_query)
        clean_query = clean_query.replace("'","")
        clean_query = clean_query.replace("<br>"," ")
        clean_query = clean_query.replace("</br>"," ")
        clean_query = clean_query.replace("<"," ")
        clean_query = clean_query.replace(">"," ")
        clean_query = clean_query.replace("?"," ")
        clean_query = clean_query.replace("-"," ")
        clean_query = clean_query.replace("."," ")
        clean_query = clean_query.replace(":"," ")
        clean_query = clean_query.replace(")"," ")
        clean_query = clean_query.replace("("," ")
        clean_query = clean_query.strip()
        clean_query = re.sub('[\\/\'":]',"",clean_query).encode( "utf-8")
        # clean_query = re.sub('[^a-zA-Z0-9\n\.]', ' ', clean_query)
        # clean_query = re.sub('[\\/\'":]',"",clean_query).encode( "ascii","ignore")
        return clean_query

    def getData(self):
        
        with open('./amazon_cells_labelled.txt') as fp:
            data = fp.readlines()
        labels = [sent.split('\t')[1] for sent in data]
        labels = [int(sent.split('\n')[0]) for sent in labels]
        labels = np.array(labels).reshape(len(labels), 1)
        documents = [self.cleanQuery(sent.split('\t')[0]) for sent in data]
        
        if self.remove_stopwords == 1:
            documents = [[word.lower() for word in sent.split() if word not in self.stopwords]
                        for sent in documents ]
            documents = [ " ".join(doc) for doc in documents]
            
        print "\n documents ",documents
        x_train, x_test, y_train, y_test = train_test_split(documents, labels, test_size = 0.3, random_state=42)

        return x_train, x_test, y_train, y_test

    
    def trainModel(self):
        x_train, x_test, y_train, y_test = self.getData()
        print "\n training size : ",len(x_train)
        train_vector = self.vectorizer.fit_transform(x_train)
        test_vector = self.vectorizer.transform(x_test)
        self.clf.fit(train_vector, y_train.ravel())
        print "\n training accuracy :: ",self.clf.score(train_vector, y_train.ravel())
        pred = self.clf.predict(test_vector)
        print "\n classification_report :: \n",classification_report(y_test, pred)
        print "\n test accuracy == ",accuracy_score(y_test, pred)
        
    def test_query(self, query):
        query = self.cleanText(query)
        print "\nquery : ",query
        query_vector = self.vectorizer.transform([query])
        print "\n pred_class = ", self.labels[self.clf.predict(query_vector)[0]]
        
    
if __name__ == '__main__':
    obj = SentimentAnalyzer()
    obj.trainModel()
    query_lst = ['i am not satisfied with the conversation','its good but can be better','your chatbot is not good','you are such a dangerous']
    for qry in query_lst:
        obj.test_query(qry)