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
from nltk.corpus import stopwords
from pandas import read_csv
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import SGDClassifier


################
# 1. LOAD DATA #
################

# read in spam sms file from computer hard drive
file_location = r'C:\Users\Windows\Dropbox\AllStuff\Spam_filter\Data\SMSSpamCollection'
sms = pd.read_csv(file_location, sep='\t', names=["label", "message"])


##############################
# 2. PREPROCESSING TEXT DATA #
##############################

# lemmatizing function - remove stop words
stop = stopwords.words('english')
def tokenizer_porter(text):
    return [words for words in text.split() if words not in stop]
        
# Add character and word numbers in text msg to check predictive accuracy via
# SVM algorithm.
sms['charnum'] = sms['message'].map(lambda text: len(text))
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
    with open(path) as csv:
        for line in csv:
            text, label = line[5:-1], line[:4]
            if (label == 'ham\t'):
                label = 'ham'
            yield text, label
            
print(next(stream_docs(path=file_location)))

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

print(get_minibatch(doc_stream, size=5))                    

#####################################################################
# 4. TRAINING MODELS VIA ONLINE ALGORITHMS AND OUT-OF-CORE LEARNING #
#####################################################################









