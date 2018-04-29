This folder contains the python scripts to convert the text into the numbers, i.e. create features from the unstructured text data.

To apply any algorithm on text, we need to to first convert this text into the features. To do that I have implemented the most widely used and popular techniques in NLP.

A. Frequency based Embedding
1. using count vectors
2. using TF-IDF vectors

B. Prediction based Embedding
1. using Word2vec embeddings (pre-trained word embeddings)


Note : To create word2vec embeddings, you need to download the pre-trained word2vec embeddings, trained on GoogleNews corpus. 
	warning : This word embeddings wod2vec model is 1.6 GB in size. 
	Link to download this word2vec model = https://code.google.com/archive/p/word2vec/
