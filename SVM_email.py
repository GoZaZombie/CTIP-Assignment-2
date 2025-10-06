import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sys

email_dataset = pd.read_csv("email_dataset.csv", encoding="utf-8")

email_dataset = email_dataset[['text', 'label']]

email_dataset['text'] = email_dataset['text'].fillna("")

label_encoder = LabelEncoder()
email_dataset['label_encoded'] = label_encoder.fit_transform(email_dataset['label'])

X_train, X_test, y_train, y_test = train_test_split(
    email_dataset['text'],
    email_dataset['label_encoded'],
    test_size=0.8,
    random_state=42,
    stratify=email_dataset['label_encoded']
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words=["subject","english"],
        max_features=20000,   
        ngram_range=(1,2)     
    )),
    ("classifier", LinearSVC())
])

pipeline.fit(X_train, y_train)

y_predictions = pipeline.predict(X_test)

print("Support Vector Machine Results (Email Spam Dataset):")
print(classification_report(y_test, y_predictions, target_names=['Ham', 'Spam']))

conf_matrix = confusion_matrix(y_test, y_predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Safe', 'Spam'], yticklabels=['Safe', 'Spam'])
plt.title("Confusion Matrix - Email Spam SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


joblib.dump(pipeline, "svm_email_spam_pipeline.pkl")



def classify_email_with_svm(email_text: str) -> str:

  
    loaded_pipeline = joblib.load("svm_email_spam_pipeline.pkl")
    
    
    predicted_class_index = loaded_pipeline.predict([email_text])[0]
    predicted_label = "Spam" if predicted_class_index == 1 else "Ham"
    
    return predicted_label



def main(args):
    if len(args) > 1:
        print(args[1]," is ", classify_email_with_svm(args[1]))
    else:
        print("invalid query")

if __name__ == "__main__":
    main(sys.argv)
