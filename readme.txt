Instructions for using this codebase:

This repo should contain the .pkl files for each model already.

If they are missing, run the following program:

LR_SMS.py
SVM_email.py
GRUTrainSpam.ipynb
NaiveBayesTrain.ipynb

TO QUERY THE MODELS:

Run CLASSIFY.py from the terminal inside the project directory:

python CLASSIFY.py <model> <input string to query>

these are the accepted inputs for <model>:

LR - Logistic Regression
SVM - Linear Vector Classifier
NB - Naive Bayes 
GRU - Gated Recurrent Unit
