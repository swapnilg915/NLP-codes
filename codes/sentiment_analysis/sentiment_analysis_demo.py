from textblob.classifiers import NaiveBayesClassifier as NBC

training_corpus = [('I am exhausted of this work','class_B'),
                  ('I cant cooperate with this','class_B'),
                  ('He is my badest enemy','class_B'),
                  ('my manager is very poor','class_B'),
                  ('I love this burger','class_A'),
                  ('This is an amazing place','class_A'),
                  ('I feel very good about these dates','class_A'),
                  ('This is my best work','class_A'),
                  ('what an awesome view','class_A'),
                  ('I dont like this dish','class_B')]

test_corpus = [('I am not feeling well today','class_B'),
              ('I feel brilliant','class_A'),
              ('Gary is a friend of mine','class_A'),
              ('I cant beleive I am doing this','class_B'),
              ('The date was good','class_A'),
              ('I do not enjoy my job','class_B')]

model_1 = NBC(training_corpus)
print(model_1.classify('their codes are amazing.'))
print(model_1.classify("I don't like their computer."))
print(model_1.accuracy(test_corpus))

#### text classification pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm

# preparing data for svm model

train_data = []
train_labels = []

for row in training_corpus:
	train_data.append(row[0])
	train_labels.append(row[1])

print "\n train_data => ",train_data
print "\n train_labels => ",train_labels

test_data = []
test_labels = []

for row in  test_corpus:
	test_data.append(row[0])
	test_labels.append(row[1])

print "\n test_data => ",test_data
print "\n test_labels => ",test_labels

#### create feature vectors
vectorizer = TfidfVectorizer(min_df = 4, max_df = 0.9)

#### train the feature vectors
train_vectors = vectorizer.fit_transform(train_data)
print "\n train_vectors => ",train_vectors

#### Apply model on test data
test_vectors = vectorizer.transform(test_data)
print "\n test_vectors => ",test_vectors

#### perform classification with svm, kernel=linear
model_2 = svm.SVC(kernel='rbf')
model_2.fit(train_vectors, train_labels)
prediction = model_2.predict(test_vectors)

print(classification_report(test_labels,prediction))


