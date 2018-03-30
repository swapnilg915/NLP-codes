import gensim
from gensim import corpora,models,similarities
from gensim.models import LsiModel, LogEntropyModel, Word2Vec
from gensim.corpora import TextCorpus, MmCorpus
from collections import defaultdict
from nltk.corpus import stopwords
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import math
import numpy as np
# from unidecode import unidecode

class TopicModeling(object):

	def __init__(self,):

		self.stemmer = PorterStemmer()
		self.lemmatizer = WordNetLemmatizer()

	def getData(self,):

		documents = []

		# documents = ['car insurance',
		# 	 'car insurance coverage',
		# 	 'auto insurance',
		# 	 'best insurance',
		# 	 'how much is the car insurance',
		# 	 'best auto coverage',
		# 	 'auto policy',
		# 	 'car policy insurance']


		data = json.load(open("modified_edel_9.json","r"))
		for dic in data['data']:
			if dic not in documents:
				
				doc = dic['text']
				doc = self.cleanText(doc)
				# doc = self.lemmatizer.lemmatize(doc)
				# doc = self.stemmer.stem(doc)

				documents.append(doc)
		documents = list(set(documents))
		return documents

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


	def preprocessing(self, query):

		documents = self.getData()
		stop_words = ([set(stopwords.words('english'))])

		if documents and query:

			texts = [[ word.lower().strip().encode('ascii','ignore') for word in document.split()
				   if word.lower() not in stop_words]
				   for document in documents]
			
			for ind,doc in enumerate(documents):
				print(ind,"::::",doc,"\n")

			## remove the word if its frequency is less than 1

			# freq = defaultdict(int)
			# for text in texts:
			# 	for token in text: 
			# 		freq[token] += 1
			# texts = [[token for token in text if freq[token] > 1]
			# 	for text in texts]


			dictionary = corpora.Dictionary(texts)
			print("\nlength of dictionary = ",len(dictionary))
			corpus = [dictionary.doc2bow(text) for text in texts] #### this is nothing but Document-Term-Matrix
			vec_bow = dictionary.doc2bow(query.lower().split())

			## log entropy operations
			# logent_transformation = LogEntropyModel(corpus,id2word=dictionary)
			# ## log entropy transformation of corpus
			# corpus = logent_transformation[corpus]
			# ## log entropy transformation of query
			# vec_bow = logent_transformation[[vec_bow]]

			## word2vec
			model = gensim.models.Word2Vec(texts, size=math.sqrt(len(dictionary)), min_count=2, window=4,sg=1, workers = 4)
			## sg = 1 => skipgram
			## sg = 2 =>  CBoW
			model.init_sims(replace=True)
			model_name = "300features_40minwords_10context"
			model.save(model_name)


			print "doesnt match = ",model.doesnt_match("salary pay allowance communication".split())
			print "\n most similar = ",model.most_similar("computer")

			# print "\n type of syn0 = ",type(model.wv.syn0)
			# print "\n shape of syn0 = ",model.wv.syn0.shape
			# raw_input("========")
			num_features = int(math.sqrt(len(dictionary)))
			# trainDataVecs = self.getAvgFeatureVecs( texts, model, num_features )
			# print "\ntrainDataVecs = ",trainDataVecs


			vocab = list(model.wv.vocab.keys())
			# print "\n vocabulary = ",vocab[:11]
			corpus_word2vec = model.wv.syn0
			word_vectors = model.similar_by_vector
			# print "\nword_vectors =  ",word_vectors
			for word in query.lower().split():
				# print "\nmodel.wv = \n",model.wv[word]
				print "\n model.wv.similarity = ",word," and ",'computer',model.wv.similarity(word, 'retirement')
			# print "\n dir(model) = ",dir(model)
			word2vec_corpus = model.wv[texts]
			# print "\n word2vec_corpus = ",word2vec_corpus

			# self.lsiModel(corpus, dictionary, vec_bow, trainDataVecs)
			# self.ldaModel(corpus, dictionary, vec_bow)


	def getAvgFeatureVecs(self, texts, model, num_features ):

		counter = 0
		reviewFeatureVecs = np.zeros((len(texts),num_features),dtype="float32")

		for text in texts:
			reviewFeatureVecs[counter] = self.makeFeatureVec(text, model, num_features)
			counter = counter + 1

		print "\n reviewFeatureVecs = ",reviewFeatureVecs
		return reviewFeatureVecs


	def makeFeatureVec(self, words, model, num_features):

		featureVec = np.zeros((num_features),dtype='float32')
		nwords=0
		index2word_set = set(model.wv.index2word)

		for word in words:
			if word in index2word_set:
				nwords = nwords + 1
				featureVec = np.add(featureVec, model[word])
		featureVec = np.divide(featureVec, nwords)

		print "\n featureVec = ",featureVec
		return featureVec

	def lsiModel(self, corpus, dictionary, vec_bow, corpus_word2vec):

		# tfidf = models.TfidfModel(corpus, normalize=True)
		# corpus = tfidf[corpus]
		# import pdb
		# pdb.set_trace()

		lsi = models.LsiModel(corpus_word2vec, id2word=dictionary,num_topics=math.sqrt(len(dictionary)))
		vec_lsi = lsi[vec_bow]
		# print("\nvec_lsi => ",vec_lsi)
		lsi_index = similarities.Similarity(corpus = lsi[corpus_word2vec], num_features=400, output_prefix="shard")
		## link = https://williambert.online/2012/05/relatively-quick-and-easy-gensim-example-code/
		lsi_index.num_best = 5
		sims_lsi = lsi_index[vec_lsi]
		# sims_lsi = sorted(enumerate(sims_lsi),key=lambda item: -item[1])
		print("\n sims_lsi => ",sims_lsi)


	def ldaModel(self, corpus, dictionary, vec_bow):

		lda = models.ldamodel.LdaModel(corpus, num_topics = 50, id2word = dictionary)
		# print(lda.print_topics(num_topics = len(dictionary),num_words = 3))

		vec_lda = lda[vec_bow]
		# print("\n vec_lda => ",vec_lda)
		lda_index = similarities.MatrixSimilarity(lda[corpus])
		# print("\n lda_index => ",lda_index)
		sims_lda = lda_index[vec_lda]
		sims_lda = sorted(enumerate(sims_lda),key=lambda item: -item[1])
		print("\n sims_lda => ",sims_lda)


if __name__ == '__main__':
	obj = TopicModeling()
	# obj.preprocessing("Documents needed to claim Medical reimbursement.")
	obj.preprocessing("Pension System")