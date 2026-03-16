#Multinomial Naive Bayes 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("clean_reviews.csv")

required = {"review text", "rating"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Mangler kolonner: {missing}. Fant: {list(df.columns)}")

df = df.dropna(subset=["review_text", "rating"]).copy()
df["review_text"] = df["review_text"].astype(str)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating"]).copy()
df["rating"] = df["rating"].astype(int)
df = df[(df["rating"] >= 1) & (df["rating"] <= 5)].copy()

X = df["review_text"]
y = df["rating"]

#Multinomial Naive Bayes is used when features represent the frequency of terms 
#(such as word counts) in a document. It is commonly applied in text classification, 
#where term frequencies are important.