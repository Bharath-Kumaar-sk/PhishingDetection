import nltk
import re
import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer as Tfv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression as LG

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_csv("Phishing_Email.csv")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    for w in tokens:
        if w not in stop_words:
            filtered_tokens.append(w)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    preprocessed_email = ' '.join(lemmatized_tokens)
    return preprocessed_email

df["Email Text"] = df["Email Text"].fillna("").astype(str)
df["Clean text"] = df["Email Text"].apply(clean_text)
print(df[["Email Text", "Clean text","Email Type"]].head(3))

vectorizer = Tfv(max_features=5000)
X = vectorizer.fit_transform(df["Clean text"])
Y = df["Email Type"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y, test_size = 0.2, random_state=42
)

#model = MultinomialNB()
#model.fit(X_train,Y_train)
#Y_pred = model.predict(X_test)

log_reg_model = LG(max_iter=1000)
log_reg_model.fit(X_train,Y_train)
log_Y_pred = log_reg_model.predict(X_test)

#print("Naive Bayes")
print("Accuracy: ", accuracy_score(Y_test,log_Y_pred))
print(classification_report(Y_test,log_Y_pred))



