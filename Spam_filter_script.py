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
print(sms_messages.head())
sms_messages.shape
                           