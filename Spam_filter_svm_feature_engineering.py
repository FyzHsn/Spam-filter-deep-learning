# -*- coding: utf-8 -*-
"""
Due to the huge dimensions of the bag-of-words, I am running into memory 
issues. In this script, I reduce the number of features via the removal of stop
words. Afterwards, I go back to looking at the effect of including additional
features such as the number of characters, words or words in all caps to the 
basic vector representation of the text. 

Furhermore, I might implement out-of-core learning as a redundancy for memory
problems.

Author: Faiyaz Hasan
Data created: September 27, 2016
"""

# import packages
import numpy as np
import pandas as pd
from scipy import sparse
from nltk.corpus import stopwords
from pandas import read_csv
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import SGDClassifier


################
# 1. LOAD DATA #
################

# sms file from computer hard drive - no need to load data
file_location = r'C:\Users\Windows\Dropbox\AllStuff\Spam_filter\Data\SMSSpamCollection'
sms = pd.read_csv(file_location, sep='\t', names=["label", "message"])

##############################
# 2. PREPROCESSING TEXT DATA #
##############################

# lemmatizing function - remove stop words
stop = stopwords.words('english')
def tokenizer(text):
    return [words for words in text.split() if words not in stop]
        
# Add character and word numbers in text msg to check predictive accuracy via
# SVM algorithm.
sms['wordnum'] = sms['message'].map(lambda text: len(text.split()))

# Add the number of words in all caps
def capwords(text):
    text_words = text.split()  
    text_words_upper_case = [word.upper() for word in text_words]
    return len(set(text_words) & set(text_words_upper_case))
        


###################################
# 3. CONVERT TEXT DATA TO VECTORS #
###################################

nrows = sum(1 for _ in open(file_location))
print(nrows)

# generator function that reads in one document at a time
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        for line in csv:
            text, label = line[4:-1], line[:4]
            if (label == 'ham\t'):
                label = 'ham'
            text = list(text) # convert string into characters
            
            if (text[0] == '\t'):
                del text[0] # remove tab character
            
            text = ''.join(text) # join characters back to text string
            yield text, label

# take a document stream from stream_docs and return a particular number of 
# documents specified by the size parameter
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

doc_stream = stream_docs(path=file_location)

#print(get_minibatch(doc_stream, size=5))                    

#####################################################################
# 4. TRAINING MODELS VIA ONLINE ALGORITHMS AND OUT-OF-CORE LEARNING #
#####################################################################

# add features to text data set and return as a vector
def addfeatures(data):
    # additional feature    
    capword_feat = np.array([capwords(lines) for lines in data])
    capword_feat.shape = (len(data), 1)
    capword_feat = sparse.csc_matrix(capword_feat)
    char_feat = np.array([len(lines) for lines in data])
    char_feat.shape = (len(data), 1)
    char_feat = sparse.csc_matrix(char_feat)
    data = vect.transform(data)
    data = sparse.hstack([char_feat, capword_feat, data])    
    return data
    
def addnofeatures(data):
    # additional feature    
    data = vect.transform(data)
    return data    

# vectorize text via HashingVectorizer
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)
clf = SGDClassifier(loss='hinge', random_state=1, n_iter=1)

classes = np.array(['ham', 'spam'])

for _ in range(40):
    X_train, y_train = get_minibatch(doc_stream, size=100)
    if not X_train:
        break        
    X_train = addnofeatures(X_train)    
    clf.partial_fit(X_train, y_train, classes=classes)
    
X_test, y_test = get_minibatch(doc_stream, size=(nrows - 5 - 4000))
X_test = addnofeatures(X_test)    

print('Accuracy: %.3f' % (clf.score(X_test, y_test)*100))    





