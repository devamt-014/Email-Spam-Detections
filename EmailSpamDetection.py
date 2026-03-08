import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC 


data = pd.read_csv('datasets/email.csv')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text

data["Message"] = data["Message"].apply(clean_text)
X = data['Message']
y = data['Category']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1,2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


logistic_model = LogisticRegression()
nbmodel = MultinomialNB()
svmmodel = LinearSVC()


models = {
    "Logistic Regression": logistic_model,
    "Naive Bayes": nbmodel,
    "SVM Model": svmmodel
}


for name, model in models.items():

    print("\n", name)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred)*100)
    print("Classification Report\n")
    print(classification_report(y_test, y_pred))

    cmf = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cmf)


pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(logistic_model, open("logistic_model.pkl", "wb"))
pickle.dump(nbmodel, open("nb_model.pkl", "wb"))
pickle.dump(svmmodel, open("svm_model.pkl", "wb"))

print("\nModels and vectorizer saved successfully.")