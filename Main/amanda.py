# Model 1: Predicting the score with KNN + classification report
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


path_in = "info284---semester-assignment/Main/reviews_scored.csv"
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
df = df[(df["score"] >= 1) & (df["score"] <= 5)].copy()

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

# Train ONLY on training split
knn_model.fit(X_train, y_train)

# Predict on test split
y_pred = knn_model.predict(X_test)

print("\n--- Classification report (TEST) ---")
print(classification_report(y_test, y_pred, digits=3))

print("\nAccuracy (TEST):", accuracy_score(y_test, y_pred))

print("\n--- Confusion matrix (TEST) ---")
print(confusion_matrix(y_test, y_pred))

# If you still want to save predictions for ALL rows in a CSV,
# retrain on ALL data AFTER evaluation:
knn_model.fit(X, y)
df["predicted_score"] = knn_model.predict(X)

cols = list(df.columns)
cols.remove("predicted_score")
score_idx = cols.index("score")
cols.insert(score_idx + 1, "predicted_score")
df = df[cols]

path_out = "(Model 1) reviews_knn.csv"
df.to_csv(path_out, index=False)

print("\nFerdig. Lagret som:", path_out)
print(df[["score", "predicted_score"]].head())