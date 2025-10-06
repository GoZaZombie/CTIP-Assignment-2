import joblib
import sys

#Calling models once here  to avoid calling multiple times 
NBModel = joblib.load("NaiveBayesModel.pkl")
NBVectorizer = joblib.load("NaiveBayesVectorizer.pkl")

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
#ADD OTHER MODELS HERE SIMILARLY

# Naive Bayes Model 
def Classify_SMS_NB(message: str, NBModel, NBVectorizer) -> tuple:
    message_tfidf = NBVectorizer.transform([message])
    
    probability = NBModel.predict_proba(message_tfidf)[0]
    predicted_class = probability.argmax()  # 0 = Safe, 1 = Spam 
    confidence = probability[predicted_class]
    
    label = "Spam" if predicted_class == 1 else "Safe"
    #confidence = (confidence * 100)
    return label, confidence


def main(args):
    if len(args) > 2:
        match args[1]:
            case ("SVM"):
                print ("Email classified using Support Vector Machine Model:")
                print("[",args[2],"] is ", classify_email_with_svm(args[2]))
            case ("LR"):
                print ("SMS message classified using Logistic Regression:")
                print("[",args[2],"] is ", classify_sms_with_lr(args[2])[0], " Confidence(%): ",classify_sms_with_lr(args[2])[1])
            case ("NB"):
                #could add an option here to select SMS/Emails? if we want to
                print ("SMS classified using Naive Bayes Classifier model:")
                label, confidence = Classify_SMS_NB(args[2], NBModel, NBVectorizer)
                
                print(f"[{args[2]}] is {label} \nConfidence(%) in this answer: {confidence:.1%}")
                
            case ("KM"):
                # add k-means output
                return
            case _:
                return "invalid model option"
    else:
        print("invalid query")

if __name__ == "__main__":
    main(sys.argv)