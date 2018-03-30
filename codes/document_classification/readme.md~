This directory contains the ready to use python scripts with the implementation of machine learning classification algorithms : 

1. Logistic regression (LR)
2. Naive Bayes (NB)
3. Support Vector Machines (SVM)
4. Random Forest (RF)

Here I have implemented all the algorithms for the purpose of binary classification (personal / question) 
class 1 => personal query - related to the user who is asking
class 2 => It is a general question. 

Though, it can be used for the multi-class classification by making few modifications to the script.
All the above classifiers are implemented using python's machine learning library - Scikit Learn. The feature extraction method used in all classifiers is a widely used Term Frequency - Inverse Docuement Frequecny. (TF-IDF)

The steps for text classification are:

1. Read the text data in python script.
2. Cleaning and preprocessing.
3. Divide the data into train / test sets. Here I have used sklearn's inbuilt cross validation method called train_test_split. 
4. Convert this text data into numbers, called as feature extraction. I have done this by using TF-IDF. 
   You can try 
   a) count vectorizer (counts the number of occurrences of each word in the data)
   b) Word2vec embeddings
   c) Part of speech (pos) tags of the words.

     as a features.

5. Train the classifier on the training data.
6. Tune the algorithm's parameters to fit well on the data. I have used sklearn's Grid search method to do this.
7. Test the accuracy of the classifier on test data. Tune the prameters till we get the required level of accuracy.
8. Along with the accuracy, we can check the classification report, which tells the precision, recall, F1 score statistics of the model.
   We can also check confusion matrix for the more information on true postives, false negatives.
9. Once we create the classifier successfully, we can use it to predict labels (class) for the new queries, using predict() function.
