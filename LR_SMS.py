import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

dataset = pd.read_csv("sms_dataset.csv", encoding="latin-1")

dataset = dataset[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text_message'})

label_encoder = LabelEncoder()
dataset['label_encoded'] = label_encoder.fit_transform(dataset['label'])

X_train, X_test, y_train, y_test = train_test_split(
    dataset['text_message'], 
    dataset['label_encoded'], 
    test_size=0.2, 
    random_state=42, 
    stratify=dataset['label_encoded']
)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

spam_classifier = LogisticRegression(max_iter=1000)
spam_classifier.fit(X_train_vectors, y_train)

y_predictions = spam_classifier.predict(X_test_vectors)
print("training results:")
print(classification_report(y_test, y_predictions, target_names=['Ham', 'Spam']))

conf_matrix = confusion_matrix(y_test, y_predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Safe', 'Spam'], yticklabels=['Safe', 'Spam'])
plt.title("Confusion Matrix - SMS Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

joblib.dump(spam_classifier, "logistic_regression_model.pkl")
joblib.dump(vectorizer, "logistic_regression_vectorizer.pkl")
print("model and vectorizer saved")






