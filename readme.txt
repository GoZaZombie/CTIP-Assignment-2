Instructions for using this codebase:

Libraries needed:
pandas
scikit-learn 
numpy 
joblib 
matplot 
seaborn 
tensorflow

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

LR - Logistic Regression (SMS)
SVM - Linear Vector Classifier (EMAIL)
NB - Naive Bayes(SMS)
NBE - Naive Bayes for (EMAIL)
GRU - Gated Recurrent Unit (EMAIL)

here's an example which uses the Naive Bayes email clasification model:

python CLASSIFY.py NBE "Subject: brighten those teeth  get your  teeth bright white now !  have you considered professional teeth whitening ? if so , you  know it usually costs between $ 300 and $ 500 from your local  dentist !"
