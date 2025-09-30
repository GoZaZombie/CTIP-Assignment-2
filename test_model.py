import joblib
def predict_message(message: str):
    # Load saved model/vectorizer
    model = joblib.load("logistic_regression_model.pkl")
    vectorizer = joblib.load("logistic_regression_vectorizer.pkl")
    
    # Transform input
    message_tfidf = vectorizer.transform([message])
    
    # Predict probabilities
    probs = model.predict_proba(message_tfidf)[0]
    predicted_class = probs.argmax()
    confidence = probs[predicted_class]
    
    label = "Spam" if predicted_class == 1 else "Safe"
    return label, confidence

# Example queries
print("Example 1:", predict_message("on the other street"))
print("Example 2:", predict_message("Thanks for your subscription to Ringtone UK your mobile will be charged å£5/month Please confirm by replying YES or NO. If you reply NO you will not be charged"))