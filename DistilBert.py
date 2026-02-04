import numpy as np
import pandas as pd
import torch
import transformers
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

wa
MODEL_NAME = 'distilbert-base-uncased'

print(f"Loading {MODEL_NAME} model and tokenizer...")
try:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertModel.from_pretrained(MODEL_NAME)
except OSError:
    print("Error: Could not load models. Please ensure you have internet access.")
    print("Run: pip install transformers torch")
    exit()

df = pd.read_csv('Phishing_Email.csv')
df["Email Text"] = df["Email Text"].fillna("").astype(str)

print("Subsampling data for speed (2000 rows)...")
df = df.sample(2000, random_state=42)

def get_bert_embeddings(text_list, batch_size=50):

    model.eval() 
    all_embeddings = []
    
    print(f"Processing {len(text_list)} emails in batches of {batch_size}...")
    
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i+batch_size]

        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(cls_embeddings)
        
        if (i // batch_size) % 5 == 0:
            print(f"  Processed batch {i // batch_size}...")
            
    return np.vstack(all_embeddings)

print("Generating BERT embeddings")
X_embeddings = get_bert_embeddings(df["Email Text"].tolist())
Y = df["Email Type"]

print("\nTraining Logistic Regression on BERT embeddings...")
X_train, X_test, Y_train, Y_test = train_test_split(X_embeddings, Y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("\n--- Transformer (DistilBERT) Results ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.4f}")
print(classification_report(Y_test, Y_pred))