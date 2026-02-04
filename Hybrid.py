import nltk
import re
import pandas as pd
import numpy as np
import gensim.downloader as api 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LG
from sklearn.metrics import accuracy_score, classification_report

print("Loading Word2Vec model...")
word2vec = api.load("word2vec-google-news-300")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('Phishing_Email.csv')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(w) for w in tokens if w not in stop_words
    ]
    return " ".join(lemmatized_tokens)

print("Cleaning text...")
df["Email Text"] = df["Email Text"].fillna("").astype(str)
df["Clean Text"] = df["Email Text"].apply(clean_text)


print("Fitting TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf.fit(df["Clean Text"])

word2weight = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

def get_weighted_vector(text, model, weights, vector_size=300):

    tokens = text.split()
    valid_vectors = []
    valid_weights = []
    
    for word in tokens:
        if word in model:
            vec = model[word]
            weight = weights.get(word, 1.0)
            valid_vectors.append(vec)
            valid_weights.append(weight)
            
    if not valid_vectors:
        return np.zeros(vector_size)

    valid_vectors = np.array(valid_vectors)
    valid_weights = np.array(valid_weights)

    weighted_sum = np.dot(valid_weights, valid_vectors)
    return weighted_sum / np.sum(valid_weights)

print("Creating weighted vectors...")
X = np.stack(df["Clean Text"].apply(lambda x: get_weighted_vector(x, word2vec, word2weight)).values)
Y = df["Email Type"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("Training Logistic Regression on Hybrid Embeddings...")
model = LG(max_iter=1000)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("\n--- Hybrid NLP Results (TF-IDF Weighted Word2Vec) ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print(classification_report(Y_test, Y_pred))