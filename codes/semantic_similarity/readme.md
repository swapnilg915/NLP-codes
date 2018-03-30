This directory contains the ready to use python scripts to find the sematic similarity for the query. 
Given a user query, these algorithms find the most similar documents to it, along with the similarity score for each document.

For this here I have implemented 3 NLP algorithms, all using the widely used python NLP package - Gensim.

1. Word Movers Distance (WMD)
	
	The WMD can also be used for the document classification. Train your WMD model on the dataset of each class. Given a new query, wmd 	    will give the similarity score for each class. The query belongs to the class for which we get the highest similarity score. (But the   	code I have put here is to use WMD for semantic similarity, not for the classification.)
	
	Note : To use WMD, you need to download the pre-trained word2vec embeddings, trained on GoogleNews corpus. 
	warning : This embeddings model is 1.6 GB in size. 
	Link to download this word2vec model = https://code.google.com/archive/p/word2vec/


2. Latent Sematic Indexing (LSI)

3. Latent Dirichlet Allocation (LDA)
	
References: 
1. Finding similar documents with Word2Vec and WMD - https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html
2. Document Similarity With Word Movers Distance - http://jxieeducation.com/2016-06-13/Document-Similarity-With-Word-Movers-Distance/
