import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack 

df = pd.read_csv('Phishing_Email.csv')
df["Email Text"] = df["Email Text"].fillna("").astype(str)

def count_patterns(text, pattern):
    return len(re.findall(pattern, text))

print("Extracting manual features...")

df['len_char'] = df['Email Text'].apply(len)
df['n_urls'] = df['Email Text'].apply(lambda x: count_patterns(x, r'http|www\.'))
df['n_ips'] = df['Email Text'].apply(lambda x: count_patterns(x, r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'))
df['n_special'] = df['Email Text'].apply(lambda x: count_patterns(x, r'[\$!%]'))
df['n_tags'] = df['Email Text'].apply(lambda x: count_patterns(x, r'<[^>]+>'))

manual_features = df[['len_char', 'n_urls', 'n_ips', 'n_special', 'n_tags']].apply(np.log1p)

print("Cleaning text for TF-IDF...")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text) 
    return text

df['Clean Text'] = df['Email Text'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['Clean Text'])

print("Combining features...")
X_combined = hstack([X_tfidf, manual_features])
Y = df["Email Type"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_combined, Y, test_size=0.2, random_state=42
)

print("Training Forensic Enhanced Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("\n--- Feature Engineering Results ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print(classification_report(Y_test, Y_pred))

feature_names = list(tfidf.get_feature_names_out()) + ['len_char', 'n_urls', 'n_ips', 'n_special', 'n_tags']
coefs = model.coef_[0]
manual_indices = range(len(feature_names) - 5, len(feature_names))

print("\nManual Feature Importance (Positive = Phishing indicator, Negative = Safe):")
for i in manual_indices:
    print(f"{feature_names[i]}: {coefs[i]:.4f}")