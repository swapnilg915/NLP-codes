from sklearn.feature_extraction.text import TfidfVectorizer

documents = ['car insurance',
			 'car insurance coverage',
			 'auto insurance',
			 'best insurance',
			 'how much is the car insurance',
			 'best auto coverage',
			 'auto policy',
			 'car policy insurance']

vect = TfidfVectorizer(lowercase=True, analyzer = 'word', use_idf=True, max_df = 1.0, min_df = 1, stop_words = 'english')
tfidf = vect.fit_transform(documents)
print tfidf


