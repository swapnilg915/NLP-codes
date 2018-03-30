import json
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import WmdSimilarity


class SemanticSimilarity(object):

	def __init__(self):
		self.w2v_data = self.load_w2v()
		self.wmd_corpus = self.getData()
	

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
		# doc = doc.replace(".","")
		doc = doc.lower().strip()
		return doc


	def word_token(self, tokens, lemma=False):
		self.stop_words = stopwords.words('english')

		tokens = tokens.encode("utf-8")
		# tokens = tokens.encode("ascii",'ignore')
		tokens = re.sub(r"([\w].)([\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\{\}\/\"\'\:\;])([\s\w].)", "\\1 \\2 \\3", tokens)
		tokens = re.sub(r"\s+", " ", tokens)
		if lemma:
			return [self.wnl.lemmatize(token) for token in word_tokenize(tokens.lower()) if token not in self.stop_words and token.isalpha()]
		else:
			return [token for token in word_tokenize(tokens.lower()) if token not in self.stop_words and token.isalpha()]

	def getData(self,):

		self.documents = []
		corpus = []

		cnt_lst = []
		data = json.load(open("data_1.json","r"))
		for dic in data['data']:
			if dic['my_id'] not in cnt_lst:
				cnt_lst.append(dic['my_id'])
				corpus.extend(sent_tokenize(dic['text']))

		corpus = list(set(corpus))
		corpus = [self.cleanText(sent) for sent in corpus if len(sent.split()) > 2]
		print "\nlen(documents) == ",len(corpus)
		self.documents = corpus
		wmd_corpus = list(map(lambda sent : self.word_token(sent), self.documents))

		return wmd_corpus

	def load_w2v(self):

		if not os.path.exists('../word2vec/GoogleNews-vectors-negative300.bin.gz'):
			raise ValueError("SKIP: You need to download the google news model")
		self.w2v_data = KeyedVectors.load_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin.gz', limit=500000, binary=True)
		return self.w2v_data

	def getWmdSimilarity(self, query):

		print "\n query ==>> ",query
		query_token = self.word_token(self.cleanText(query))
		instance_wmd = WmdSimilarity(self.wmd_corpus, self.w2v_data)
		wmd_sims = instance_wmd[query_token]
		wmd_sims = sorted(enumerate(wmd_sims), key=lambda item: -item[1])
		similar_docs = [(s, self.documents[i])  for i,s in wmd_sims]
		similar_docs = similar_docs[:5]
		print "\nsimilar docs => ",similar_docs


if __name__ == '__main__':

	obj = SemanticSimilarity()
	obj.getWmdSimilarity("what is the eligibility criteria for learning assistance program")
