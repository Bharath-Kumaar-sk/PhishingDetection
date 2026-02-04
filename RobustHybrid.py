import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

print("Loading Data...")
df = pd.read_csv('Phishing_Email.csv')
df["Email Text"] = df["Email Text"].fillna("").astype(str)

print(f"Original Row Count: {len(df)}")
df = df[df['Email Text'].str.len() > 10]
print(f"Cleaned Row Count: {len(df)}")

def count_patterns(text, pattern):
    return len(re.findall(pattern, text))

print("Extracting Manual Features...")
df['len_char'] = df['Email Text'].apply(len)
df['n_urls'] = df['Email Text'].apply(lambda x: count_patterns(x, r'http|www\.'))
df['n_ips'] = df['Email Text'].apply(lambda x: count_patterns(x, r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'))
df['n_special'] = df['Email Text'].apply(lambda x: count_patterns(x, r'[\$!%]'))
df['n_tags'] = df['Email Text'].apply(lambda x: count_patterns(x, r'<[^>]+>'))

manual_features = df[['len_char', 'n_urls', 'n_ips', 'n_special', 'n_tags']].apply(np.log1p)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

df['Clean Text'] = df['Email Text'].apply(clean_text)
print("Vectorizing Text (Words + Characters)...")

word_vectorizer = TfidfVectorizer(
    max_features=5000, 
    stop_words='english', 
    ngram_range=(1, 2) 
)
X_word = word_vectorizer.fit_transform(df['Clean Text'])

char_vectorizer = TfidfVectorizer(
    max_features=5000, 
    analyzer='char_wb', 
    ngram_range=(3, 5) 
)
X_char = char_vectorizer.fit_transform(df['Clean Text'])

print("Combining Features...")
X_combined = hstack([X_word, X_char, manual_features])
Y = df["Email Type"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_combined, Y, test_size=0.2, random_state=42
)

print("Training Final Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("\n--- Final Robust Model Results ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print(classification_report(Y_test, Y_pred))

def predict_new_email(raw_text):

    demo_df = pd.DataFrame({'Email Text': [raw_text]})

    demo_df['len_char'] = demo_df['Email Text'].apply(len)
    demo_df['n_urls'] = demo_df['Email Text'].apply(lambda x: count_patterns(x, r'http|www\.'))
    demo_df['n_ips'] = demo_df['Email Text'].apply(lambda x: count_patterns(x, r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'))
    demo_df['n_special'] = demo_df['Email Text'].apply(lambda x: count_patterns(x, r'[\$!%]'))
    demo_df['n_tags'] = demo_df['Email Text'].apply(lambda x: count_patterns(x, r'<[^>]+>'))
 
    demo_manual = demo_df[['len_char', 'n_urls', 'n_ips', 'n_special', 'n_tags']].apply(np.log1p)

    demo_df['Clean Text'] = demo_df['Email Text'].apply(clean_text)

    demo_word_vec = word_vectorizer.transform(demo_df['Clean Text'])
    demo_char_vec = char_vectorizer.transform(demo_df['Clean Text'])

    demo_combined = hstack([demo_word_vec, demo_char_vec, demo_manual])
 
    prediction = model.predict(demo_combined)[0]
    probability = model.predict_proba(demo_combined).max()
    
    return prediction, probability

print("\n--- LIVE TEST RESULTS ---")

fake_1 = "URGENT!!! Your P a y P a l account is suspended. Click http://10.0.0.5/login to verify now $$$."
pred, prob = predict_new_email(fake_1)
print(f"Email: '{fake_1}'\nVerdict: {pred} ({prob:.2%})\n")

fake_2 = "Hi there, here is the weekly newsletter. check out our website at www.google.com for updates. Thanks."
pred, prob = predict_new_email(fake_2)
print(f"Email: '{fake_2}'\nVerdict: {pred} ({prob:.2%})\n")

fake_3 = "Dear customer, we noticed a login attempt. Please verify at http://secure-bank-login.com/verify"
pred, prob = predict_new_email(fake_3)
print(f"Email: '{fake_3}'\nVerdict: {pred} ({prob:.2%})\n")