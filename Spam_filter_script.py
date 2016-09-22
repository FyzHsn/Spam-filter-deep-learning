# -*- coding: utf-8 -*-
"""
Title: Spam_filter_script
Author: Faiyaz Hasan
Date created: September 22, 2016

In this script, I am interested in constructing a basic spam filter and a more 
complex one using deep learning. This will be my introduction to:
1. Natural language processing techniques and relevant python packages
2. Multiple layer neural network
3. Deep learning

"""

################################
# 1. EXPLORATORY DATA ANALYSIS #
################################

"""
Read spam file from location - the data come from the website
http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

"""
file_location = r'C:\Users\Windows\Dropbox\AllStuff\Spam_filter\Data\SMSSpamCollection'

import pandas as pd
from pandas import read_csv

sms_messages = pd.read_csv(file_location, sep='\t', names=["label", "message"])
print(sms_messages.shape)

# aggregate statistics of dataframe
print(sms_messages.groupby('label').describe())

# number of characters and words in the messages
sms_messages['char_num'] = \
            sms_messages['message'].map(lambda text: len(text))

sms_messages['word_num'] = \
            sms_messages['message'].map(lambda text: len(text.split()))                            

print(sms_messages.head())

# plot length of characters and number of words for ham and spam.
import matplotlib.pyplot as plt
sms_messages.hist(column='char_num', by='label', bins=50)
sms_messages.hist(column='word_num', by='label', bins=50)

