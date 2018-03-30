"""
Author : Swapnil Gaikwad
Title : to get the topics from the documents using Latent Dirichlet Allocation
tools : Gensim
"""

doc1 = "sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "my father spends a lot of time driving my sister around to dance practice."
doc3 = "doctors suggest that driving may cause increased stress and blood pressure."

doc_complete = [doc1,doc2,doc3]

doc_clean = [doc.split() for doc in doc_complete]
print("\n doc_clean => ",doc_clean)

import gensim 
from gensim import corpora

dictionary = corpora.Dictionary(doc_clean)
print("\n dictionary => ",dictionary)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
print("\n doc_term_matrix => ",doc_term_matrix)

Lda = gensim.models.ldamodel.LdaModel

ldamodel = Lda(doc_term_matrix, num_topics = 3, id2word = dictionary, passes = 50)

topics = ldamodel.print_topics()

print("\n topics => ",topics)
 
