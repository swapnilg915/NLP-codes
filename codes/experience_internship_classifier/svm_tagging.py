from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
import re
import traceback

# from spacy.en import English
# parser = English()

import spacy
parser = spacy.load('en')
# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", '"','`',"'ve"]


# Every step in a pipeline needs to be a "transformer". 
# Define a custom transformer to clean text using spaCy

class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """
    def transform(self, X, **transform_params):
        print "\n transform = ",transform_params
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
    

# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")
    
    # replace twitter @mentions
    mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    text = mentionFinder.sub("@MENTION", text)
    
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    
    # lowercase
    text = text.lower()

    return text


# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(sample):
    
    print "\n sample = ",sample
    # get the tokens using spaCy
    tokens = parser(sample)
    print "\n tokens 11 = ",tokens
    import pdb
    pdb.set_trace()
    # # lemmatize
    # lemmas = []
    # for i,tok in enumerate(tokens):
    #     print "\n tok = ",tok
    #     print "\n tok lemma = ",tok.lemma_, tok.lemma
    #     lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    # tokens = lemmas
    # print "\n tokens 22 = ",tokens
    
    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    print "\n tokens 33 = ",tokens
    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
    
    print "\n tokens 44 = ",tokens
    return tokens


def printNMostInformative(vectorizer, clf, N):
    
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print"Class 1 best: "
    for feat in topClass1:
        print feat
    print"Class 2 best: "
    for feat in topClass2:
        print feat
        
    
# the vectorizer and classifer to use
# note that I changed the tokenizer in CountVectorizer to use a custom function using spaCy's tokenizer

vectorizer = CountVectorizer(tokenizer=tokenizeText,min_df=1)
clf = LinearSVC()

# the pipeline to clean, tokenize, vectorize, and classify
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])

train = ["I love space. Space is great.", "Planets are cool. I am glad they exist in space", 
        "lol @twitterdude that is gr8", "twitter &amp; reddit are fun.", 
        "Mars is a planet. It is red.", "@Microsoft: y u skip windows 9?", 
        "Rockets launch from Earth and go to other planets.", "twitter social media &gt; &lt;", 
        "@someguy @somegirl @twitter #hashtag", "Orbiting the sun is a little blue green planet."]
labelsTrain = ["space", "space", "twitter", "twitter", "space", "twitter", "space", "twitter", "twitter", "space"]

test = ["i h8 riting comprehensibly #skoolsux", "planets and stars and rockets and stuff"]
labelsTest = ["twitter", "space"]

print "\n len(train) = ",len(train),train,labelsTrain

# import pdb; pdb.set_trace()

try:
    # train
    pipe.fit(train, labelsTrain)
    print "\n fitted pipe"
    # test
    preds = pipe.predict(test)
    print "\n preds = ",preds
    
    
    print"="*50
    print"results:"
    for (sample, pred) in zip(test, preds):
        print(sample, ":", pred)
    print"accuracy:", accuracy_score(labelsTest, preds)
    
    
    print"="*50
    print"Top 10 features used to predict: "
    # show the top features
    printNMostInformative(vectorizer, clf, 10)
    
    
    print"="*50
    print"The original data as it appeared to the classifier after tokenizing, lemmatizing, stoplisting, etc"
    # let's see what the pipeline was transforming the data into
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
    transform = pipe.fit_transform(train, labelsTrain)
    
    # get the features that the vectorizer learned (its vocabulary)
    vocab = vectorizer.get_feature_names()

except Exception,e:
    print "\nException occurred = ", e,"\n"
    print traceback.format_exc()
# the values from the vectorizer transformed data (each item is a row,column index with value as # times occuring in the sample, stored as a sparse matrix)
for i in range(len(train)):
    s = ""
    indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
    numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
    for idx, num in zip(indexIntoVocab, numOccurences):
        s += str((vocab[idx], num))
    print "Sample {}: {}".format(i, s)