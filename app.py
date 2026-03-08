import streamlit as st
import pickle
import re


vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
logistic_model = pickle.load(open("logistic_model.pkl", "rb"))
nb_model = pickle.load(open("nb_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))

st.title("Email Spam Detection Web-App")

user_message = st.text_area("Enter a message")

# Model selection
model_choice = st.selectbox(
    "Choose Model",
    ("Logistic Regression", "Naive Bayes", "SVM")
)

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text


if st.button("Run Model"):

    # Clean user message first
    cleaned_message = clean_text(user_message)

    # Vectorize user message
    user_vec = vectorizer.transform([cleaned_message])

    if model_choice == "Logistic Regression":
        prediction = logistic_model.predict(user_vec)

    elif model_choice == "Naive Bayes":
        prediction = nb_model.predict(user_vec)

    else:
        prediction = svm_model.predict(user_vec)

    st.subheader("Prediction")

    if prediction[0] == "spam":
        st.error("🚨 This message is SPAM")
    else:
        st.success("✅ This message is NOT SPAM")