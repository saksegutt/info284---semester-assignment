# Model 1: Predicting the rating with K-Nearest Neighbour + classification report

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("clean_reviews.csv")

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

df = df[df["rating"] != 3].copy()
df["rating_binary"] = df["rating"].apply(lambda x: 1 if x >= 4 else 0)
print(df["rating_binary"].value_counts())

X = df["review_text"]
y = df["rating_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,stratify=y
)

for k in [3, 5, 7, 9]:
    for ngram in [(1,1), (1,2)]:
        knn_model = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=ngram, max_features=30000)),
            ("knn", KNeighborsClassifier(n_neighbors=k, metric="cosine"))
        ])
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"k={k}, ngram={ngram}, Accuracy={acc:.3f}")

knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

print("\n--- Classification report (TEST) ---")
print(classification_report(y_test, y_pred, target_names=["Negative","Positive"]))
print("\nAccuracy (TEST):", accuracy_score(y_test, y_pred))
print("\n--- Confusion matrix (TEST) ---")
print(confusion_matrix(y_test, y_pred))


