# 📧 Email Spam Detection Web App

A Machine Learning project that detects whether an email message is **Spam or Not Spam**.
The project includes a trained model and a **Streamlit web application** where users can input messages and test different ML models.

---

## 🚀 Features

* Spam detection using **Machine Learning**
* Three classification models:

  * Logistic Regression
  * Naive Bayes
  * Support Vector Machine (SVM)
* Text preprocessing and TF-IDF vectorization
* Interactive **Streamlit web interface**
* Model comparison capability

---

## 🧠 Models Used

The following models were trained and evaluated:

| Model                  | Description                                     |
| ---------------------- | ----------------------------------------------- |
| Logistic Regression    | Linear model commonly used for classification   |
| Naive Bayes            | Probabilistic classifier suitable for text data |
| Support Vector Machine | Powerful classifier for high-dimensional data   |

---

## 📂 Project Structure

```
Email-Spam-Detection
│
├── datasets/
│   └── email.csv
│
├── EmailSpamDetection.py   # Training script
├── app.py                  # Streamlit web app
│
├── vectorizer.pkl          # TF-IDF vectorizer
├── logistic_model.pkl
├── nb_model.pkl
└── svm_model.pkl
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/yourusername/email-spam-detection.git
cd email-spam-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Web App

```
streamlit run app.py
```

Then open the browser and test messages with different models.

---

## 💡 Example Test Messages

Spam example:

```
Congratulations! You have won a free prize. Claim your reward now!
```

Normal message:

```
Hey, are we still meeting for lunch today?
```

---

## 🛠 Technologies Used

* Python
* scikit-learn
* pandas
* Streamlit
* TF-IDF Vectorization

---

## 📌 Future Improvements

* Better NLP preprocessing
* Model probability scores
* Visualization of model comparison
* Deployment of the web app

---

## 👨‍💻 Author

Devam Trivedi
AI & Machine Learning Student
