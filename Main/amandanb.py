#Multinomial Naive Bayes 
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

path_in = "reviews_scored.csv"

if not os.path.exists(path_in):
    path_in = "clean_reviews.csv"

df = pd.read_csv(path_in)

required = {"review_text", "rating"}
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if y.nunique() > 1 else None
)

nb_model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=30000)),
    ("nb", MultinomialNB())
])

# Train
nb_model.fit(X_train, y_train)

# Predict
y_pred = nb_model.predict(X_test)

print("\n--- Classification report (TEST) ---")
print(classification_report(y_test, y_pred, digits=3))

print("\nAccuracy (TEST):", accuracy_score(y_test, y_pred))

print("\n--- Confusion matrix (TEST) ---")
print(confusion_matrix(y_test, y_pred))

# Retrain on full dataset for saving predictions
nb_model.fit(X, y)
df["predicted_score"] = nb_model.predict(X)

cols = list(df.columns)
cols.remove("predicted_score")
score_idx = cols.index("rating")
cols.insert(score_idx + 1, "predicted_score")
df = df[cols]

path_out = "(Amanda) reviews_naive_bayes.csv"
df.to_csv(path_out, index=False)

print("\nFerdig. Lagret som:", path_out)
print(df[["rating", "predicted_score"]].head())

#Multinomial Naive Bayes is used when features represent the frequency of terms 
#(such as word counts) in a document. It is commonly applied in text classification, 
#where term frequencies are important.