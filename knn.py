import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

path_in = "reviews_scored.csv"
if not os.path.exists(path_in):
    path_in = "reviews_scored.csv"

df = pd.read_csv(path_in)

required = {"review_text", "score"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Mangler kolonner: {missing}. Fant: {list(df.columns)}")

df = df.dropna(subset=["review_text", "score"]).copy()
df["review_text"] = df["review_text"].astype(str)
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna(subset=["score"]).copy()
df["score"] = df["score"].astype(int)
df = df[(df["score"] >= 1) & (df["score"] <= 10)].copy()

X = df["review_text"]
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if y.nunique() > 1 else None
)

knn_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=30000
    )),
    ("knn", KNeighborsClassifier(
        n_neighbors=7,
        metric="cosine"
    ))
])

knn_model.fit(X_train, y_train)

knn_model.fit(X, y)
df["predicted_score"] = knn_model.predict(X)

cols = list(df.columns)
cols.remove("predicted_score")
score_idx = cols.index("score")
cols.insert(score_idx + 1, "predicted_score")
df = df[cols]

path_out = "reviews_knn.csv"
df.to_csv(path_out, index=False)

print("Ferdig. Lagret som:", path_out)
print(df[["score", "predicted_score"]].head())