from gensim import corpora,models,similarities
from collections import defaultdict
from nltk.corpus import stopwords

class TopicModeling(object):

	def __init__(self,):
		pass

	def getData(self,):

		documents = ['car insurance',
			 'car insurance coverage',
			 'auto insurance',
			 'best insurance',
			 'how much is the car insurance',
			 'best auto coverage',
			 'auto policy',
			 'car policy insurance']

		return documents


	def preprocessing(self, query):

		documents = self.getData()
		stop_words = ([set(stopwords.words('english'))])

		if documents and query:

			texts = [[ word.lower() for word in document.split()
				   if word.lower() not in stop_words]
				   for document in documents]

			print("\n texts => ",texts)

			freq = defaultdict(int)

			for text in texts:
				print("\n text => ",text)
				for token in text: 
					print("\n token => ",token)
					freq[token] += 1

			print("\n freq 2 => ",freq)

			texts = [[token for token in text if freq[token] > 1]
				for text in texts]

			print("\n texts with token > 1 => ",texts)

			dictionary = corpora.Dictionary(texts)
			print("\n dictionary => ",dictionary)

			corpus = [dictionary.doc2bow(text) for text in texts] #### this is nothing but Document-Term-Matrix
			print("\n corpus => ",corpus)

			vec_bow = dictionary.doc2bow(query.lower().split())
			self.lsiModel(corpus, dictionary, vec_bow)
			self.ldaModel(corpus, dictionary, vec_bow)


	def lsiModel(self, corpus, dictionary, vec_bow):
	
		lsi = models.LsiModel(corpus,id2word=dictionary,num_topics=len(vec_bow))

		vec_lsi = lsi[vec_bow]
		print("\nvec_lsi => ",vec_lsi)
		lsi_index = similarities.MatrixSimilarity(lsi[corpus])
		print("\n lsi_index => ",lsi_index)
		sims_lsi = lsi_index[vec_lsi]
		sims_lsi = sorted(enumerate(sims_lsi),key=lambda item: -item[1])
		print("\n sims_lsi => ",sims_lsi)


	def ldaModel(self, corpus, dictionary, vec_bow):

		lda = models.ldamodel.LdaModel(corpus, num_topics = len(vec_bow), id2word = dictionary, passes = 50)
		print(lda.print_topics(num_topics = len(vec_bow),num_words = 3))

		vec_lda = lda[vec_bow]
		print("\n vec_lda => ",vec_lda)
		lda_index = similarities.MatrixSimilarity(lda[corpus])
		print("\n lda_index => ",lda_index)
		sims_lda = lda_index[vec_lda]
		print("\n sims_lda => ",sims_lda)


if __name__ == '__main__':
	obj = TopicModeling()
	obj.preprocessing("i want to buy a best car insurance.")