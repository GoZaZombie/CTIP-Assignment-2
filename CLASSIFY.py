import joblib
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Calling models once here  to avoid calling multiple times 
NBModel = joblib.load(r"ModelTraining/NaiveBayesModel.pkl")
NBVectorizer = joblib.load(r"ModelTraining/NaiveBayesVectorizer.pkl")
GRUModel = load_model(r"ModelTraining/GRUModelEmail.h5")
tokenizer = joblib.load(r"ModelTraining/tokenizer.pkl")
NBEModel = joblib.load(r"ModelTraining/NaiveBayesModelEmail.pkl")
NBEVectorizer = joblib.load(r"ModelTraining/NaiveBayesVectorizer2.pkl")




#___SVM_MODEL___
def classify_email_with_svm(email_text: str) -> str:

    loaded_pipeline = joblib.load("svm_email_spam_pipeline.pkl")
    
    predicted_class_index = loaded_pipeline.predict([email_text])[0]
    predicted_label = "Spam" if predicted_class_index == 1 else "Safe"
    
    return predicted_label
#___LOGISTIC_REGRESSION_MODEL___
def classify_sms_with_lr(message: str) -> str:

    model = joblib.load("logistic_regression_model.pkl")
    vectorizer = joblib.load("logistic_regression_vectorizer.pkl")
    
    message_tfidf = vectorizer.transform([message])
    
    probs = model.predict_proba(message_tfidf)[0]
    predicted_class = probs.argmax()
    confidence = probs[predicted_class]
    
    label = "Spam" if predicted_class == 1 else "Safe"
    return label, confidence
# Naive Bayes Model SMS
def Classify_SMS_NB(message: str) -> str:
    message_tfidf = NBVectorizer.transform([message])
    
    probability = NBModel.predict_proba(message_tfidf)[0]
    predicted_class = probability.argmax()
    confidence = probability[predicted_class]
    
    label = "Spam" if predicted_class == 1 else "Safe"
    
    return label, confidence

def Classify_EMAIL_NB(message: str) -> str:
    message_tfidf = NBEVectorizer.transform([message])
    
    probability_E = NBEModel.predict_proba(message_tfidf)[0]
    predicted_class_E = probability_E.argmax()  # 0 = Safe, 1 = Spam 
    confidence_E = probability_E[predicted_class_E]
    
    label = "Spam" if predicted_class_E == 1 else "Safe"
    
    return label, confidence_E
#GRU Model
def classify_Email_GRU(message: str) -> str:
    maxlen=100
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=maxlen, padding='post')

    probability = GRUModel.predict(padded, verbose=0)[0][0]
    
    label = "Spam" if probability > 0.5 else "Safe"
    confidence = probability if label == "Spam" else 1 - probability
    
    return label, confidence


def run_model_classification(message: str, ModelSelection) -> tuple: 
    Model_function_dic = {
    "NBE" : Classify_EMAIL_NB,
    "NBSMS" : Classify_SMS_NB, 
    "GRU" : classify_Email_GRU, 
    "SVM" : classify_email_with_svm,
    "LR" : classify_sms_with_lr
    }
    if ModelSelection in Model_function_dic: 

        Selected_function = Model_function_dic[ModelSelection]

        return Selected_function(message)[0], Selected_function(message)[1] if len(Selected_function(message)) > 1 else None
    else: 
        print(f"Invalid Model Selection")
        return 0 


def main(args):
    if len(args) > 2:
        match args[1]:
            case ("SVM"):
                print ("Email classified using Support Vector Machine Model:")
                print("[",args[2],"] is ", classify_email_with_svm(args[2]))
            case ("LR"):
                print ("SMS message classified using Logistic Regression:")
                print("[",args[2],"] is ", classify_sms_with_lr(args[2])[0], " Confidence(%): ",classify_sms_with_lr(args[2])[1])
            case ("NBSMS"):
                print ("SMS classified using Naive Bayes Classifier model:")
                label, confidence = Classify_SMS_NB(args[2])
                print(f"[{args[2]}] is {label} \nConfidence(%) in this answer: {confidence:.1%}")
            case("NBE"): 
                print ("Email classified using Naive Bayes Classifier model:")
                label, confidence = Classify_EMAIL_NB(args[2])
                print(f"[{args[2]}] is {label} \nConfidence(%) in this answer: {confidence:.1%}")
            case ("GRU"):
                print ("Email classified using GRU Model:") 
                label, confidence = classify_Email_GRU(args[2])
                print(f"[{args[2]}] is {label} \nConfidence(%) in this answer: {confidence:.1%}") #can remove confidence values, just added them since both models give a score between 0-1
            case _:
                return "invalid model option"
    else:
        print("invalid query")

if __name__ == "__main__":
    main(sys.argv)