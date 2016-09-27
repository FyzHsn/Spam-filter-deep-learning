Spam-filter-deep-learning
=========================

Project objective
-----------------

* Train machine learning algorithm to distinguish between spam and ham text messages that are pre-labelled.   
* Learn NLP, naive bayes and deep learning algorithms/packages in Python in a practical context.   

Dataset 
-------

The data set has dimensions (5572, 2). The first column called "label" takes the values "spam" or "ham" while the second column names "message" simply contains the sms message as a string. There are 5572 such messages.  

4825 ham labels vs 747 spam labels. Remember to take into account that there are many more ham than spam messages when using cross validation to determine performance.

Stages of algorithm development
-------------------------------

1. Train SVM, Logistic regression algorithms and optimize parameters for bag-of-words combined with downweighting frequently occuring words via tf-idf approach. DONE. CV accuracy: 96.3 +/- 0.5 for Logistic regression. 98.0 +/- 0.3 for Linear SVM with regularization constant C=1.0 for L2 norm. 
Using n-gram = 2, SVM - 98.4 +/- 0.3.
Using n-gram = 3, SVM - 98.7 +/- 0.3.
Using n-gram = 4, SVM - 98.5 +/- 0.4.

Hence, n-gram = 3 is optimal.

2. Next, try adding additional features such as (i) number of words in msg, (ii) number of characters, (iii) number of words in all caps to check if it leads to performance improvement. Optimize feature space.    
3. Do a grid parameter search to find optimally performing parameter on feature space.   
4. Apply multiple layer neural net and deep learning.   

Note
----

1. Simple NLP approach - create a bag of words.   
2. To keep an understanding of the order of words one can use n-grams.   
3. In addition to the words from the sms, what is the end performance if I add another feature column which shows the proportion of words that are all upper case. Spams seem to have a lot of words like "WINNER!!". What if I added the number of words in the messages? Or the number of characters?   
4. Think of the process above as dimension picking for performance optimization. Using PCA and LDA we reduced the number of dimensions to describe the data. Furthermore, think of the message as an element in a vector space. Then, the feature columns we use is equivalent to picking the approximate representation in a particular basis.  
5. It is not always wise to reduce the number of dimensions.   
6. Another lesson: If adding more data points leads to better performance of the training and the test data set, that is an indication that the model is not complex enough to capture all the information. In that setting, we can add more data or increase the complexity of the algorithm. Now, I am thinking, as in this case here, perhaps adding additional features could be the answer. I'll have to play with this idea to find out.   

References
==========

* Machine Learning Techniques in Spam Filtering by Konstantin Tretyakov.  
* Python Machine Learning by Sebastian Raschka.   
* http://radimrehurek.com/data_science_python/   


