import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

df = pd.read_csv('Phishing_Email.csv')
df["Email Text"] = df["Email Text"].fillna("").astype(str)

def count_patterns(text, pattern):
    return len(re.findall(pattern, text))

df['len_char'] = df['Email Text'].apply(len)
df['n_urls'] = df['Email Text'].apply(lambda x: count_patterns(x, r'http|www\.'))
df['n_ips'] = df['Email Text'].apply(lambda x: count_patterns(x, r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'))
df['n_special'] = df['Email Text'].apply(lambda x: count_patterns(x, r'[\$!%]'))
df['n_tags'] = df['Email Text'].apply(lambda x: count_patterns(x, r'<[^>]+>'))
manual_features = df[['len_char', 'n_urls', 'n_ips', 'n_special', 'n_tags']].apply(np.log1p)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text

df['Clean Text'] = df['Email Text'].apply(clean_text)
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(df['Clean Text'])
X_combined = hstack([X_tfidf, manual_features])
Y = df["Email Type"]

X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
    X_combined, Y, df.index, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

results = pd.DataFrame({
    'Actual': Y_test,
    'Predicted': Y_pred,
    'Original_Text': df.loc[idx_test, 'Email Text']
})

errors = results[results['Actual'] != results['Predicted']]

print(f"Total Errors: {len(errors)} out of {len(Y_test)} test samples")
print("\n--- Inspecting First 5 False Negatives ---")
print("(Actual: Phishing, but Model thought Safe)")
false_negatives = errors[errors['Actual'] == 'Phishing Email'].head(5)
for i, row in false_negatives.iterrows():
    print(f"\nExample {i}:")
    print(f"Text snippet: {row['Original_Text'][:200]}...")

print("\n--- Inspecting First 5 False Positives ---")
print("(Actual: Safe, but Model thought Phishing)")
false_positives = errors[errors['Actual'] == 'Safe Email'].head(5)
for i, row in false_positives.iterrows():
    print(f"\nExample {i}:")
    print(f"Text snippet: {row['Original_Text'][:200]}...")