# -*- coding: utf-8 -*-
"""
Title: Spam_filter_script
Author: Faiyaz Hasan
Date created: September 22, 2016

In this script, I am interested in constructing a basic spam filter and a more 
complex one using deep learning. This will be my introduction to:
1. Natural language processing techniques and relevant python packages
2. Naive Bayesian method
3. Multiple layer neural network + Deep learning

"""
######################
# 0. IMPORT PACKAGES #
######################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

################################
# 1. EXPLORATORY DATA ANALYSIS #
################################

"""
Read spam file from location - the data came from the website
http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection.
I saved it on my hard drive and loaded it from there.

feature columns for the data frame "sms" are:
label - spam or ham.
message - corresponding text messages

"""
file_location = r'C:\Users\Windows\Dropbox\AllStuff\Spam_filter\Data\SMSSpamCollection'
sms = pd.read_csv(file_location, sep='\t', names=["label", "message"])
print(sms.shape)

# aggregate statistics of dataframe
print(sms.groupby('label').describe())

## number of characters and words in the messages
#sms['char_num'] = sms['message'].map(lambda text: len(text))
#sms['word_num'] = sms['message'].map(lambda text: len(text.split()))                            

print(sms.head(4))
print(sms.message[3])

# plot length of characters and number of words for ham and spam.
#sms.hist(column='char_num', by='label', bins=30)
#sms.hist(column='word_num', by='label', bins=30)

#############################################
# 2. PROCESSING TEXT DOCUMENTS INTO VECTORS #
#############################################

## Test data frame
#columns = ['message']
#index = np.arange(3)
#sms = pd.DataFrame(columns=columns, index=index)         
#sms['message'] = np.array([["The sun is shining"], ["The weather is sweet"],
#                  ["The sun is shining and the weather is sweet"]])
         
# processing documents into tokens
def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

lancaster = LancasterStemmer()
def tokenizer_lancaster(text):
    return [lancaster.stem(word) for word in text.split()]
                  
# construct bag-of-words (bow) and write messages in terms of the
# dictionary words. n-gram range changes the number of words for each
# dictionary element. Kanaris, Houvardas, and Stamatatos found that
# n-gram choices of 3 and 4 yield good performance for anti-spam 
# filters.
count = CountVectorizer(analyzer=tokenizer, ngram_range=(1,1))
bag = count.fit_transform(sms['message'])

print('Number of words in bow: ', len(count.vocabulary_))
print('First 5 words of dictionary: ', 
      list(count.vocabulary_.keys())[0:5])

# normalizing stop words via term frequency - inverse document
# frequency using a built in function from sklearn
tfidf = TfidfTransformer()
sms_tfidf = tfidf.fit_transform(bag)

# Looking at individual sms's we see that we do not need to do any
# cleaning up of texts.


####################################################
# 3. TRAINING A MODEL VIA NAIVE BAYESIAN ALGORITHM #
####################################################

#X_train = sms_tfidf.loc[:2500].values
#y_train = sms.loc[:2500, 'label'].values
#X_test = sms_tfidf.loc[2500:, 'message'].values
#y_test = sms.loc[2500:, 'label'].values

#spam_detector = GaussianNB().fit(sms_tfidf, sms.loc[:, 'label'].values)
def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

X_train, X_test, y_train, y_test = \
    train_test_split(sms['message'], sms['label'], test_size=0.3)

lr_pipe = Pipeline([
    ('bag', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(random_state=0))])
    
scores = cross_val_score(lr_pipe,
                         X_train,
                         y_train,
                         cv=5,
                         scoring='accuracy')
                         
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores)*100, np.std(scores)*100))                         
















