# -*- coding: utf-8 -*-
"""
Due to the huge dimensions of the bag-of-words, I am running into memory 
issues. In this script, I reduce the number of features via the removal of stop
words. Afterwards, I go back to looking at the effect of including additional
features such as the number of characters, words or words in all caps to the 
basic vector representation of the text. 

Author: Faiyaz Hasan
Data created: September 27, 2016

"""

# import packages
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC

# read in spam sms file from computer hard drive
file_location = r'C:\Users\Windows\Dropbox\AllStuff\Spam_filter\Data\SMSSpamCollection'
sms = pd.read_csv(file_location, sep='\t', names=["label", "message"])

# lemmatizing function
def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

# Add character and word numbers in text msg to check predictive accuracy via
# SVM algorithm.
sms['charnum'] = sms['message'].map(lambda text: len(text))
sms['wordnum'] = sms['message'].map(lambda text: len(text.split()))

