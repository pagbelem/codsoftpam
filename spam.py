#importation des bibliotheques necessaire
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('spam.csv', encoding='latin-1')

data = data[['v1', 'v2']]
data.head

data.columns = ['label', 'message']


data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# separation des données
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.3, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
nb_predictions = nb_classifier.predict(X_test_tfidf

lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train_tfidf, y_train)
lr_predictions = lr_classifier.predict(X_test_tfidf)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)
svm_predictions = svm_classifier.predict(X_test_tfidf)

def evaluate_model(predictions, y_test, model_name):
    print(f"Results for {model_name}:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy_score(y_test, predictions)}\n")

# Évaluer chaque modèle
evaluate_model(nb_predictions, y_test, "Naive Bayes")
evaluate_model(lr_predictions, y_test, "Logistic Regression")
evaluate_model(svm_predictions, y_test, "SVM")
