"""
Author : Swapnil Gaikwad
Title : To calculate average of the word's vector using pre-trained word2vec model
tools : Gensim
features used => word2vec
Description : The word's vector or average of that vector can be used as a feature for training of machine learning algorithms, in many applications
"""

from gensim.models.keyedvectors import KeyedVectors 

word_vectors = KeyedVectors.load_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin.gz',binary=True)
vector = word_vectors['reimbursement']
print vector

''' 
average of word vectors for a particular word 
This average can be used as a feature for training any algorithm

'''
avg = 0
add = 0
for n in vector:
	add += n 
avg = add / len(vectors)
print "\n avg ==>> ",avg
