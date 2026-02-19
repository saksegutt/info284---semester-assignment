import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, mean_absolute_error

# ----------------------------
# 1) Load data
# ----------------------------
path_in = "Datacleaned_reviews.csv"  # <- endre hvis nødvendig
df = pd.read_csv(path_in)

# Forventer kolonnene: review_text og score
# (hvis score heter noe annet, endre her)
df = df.dropna(subset=["review_text", "score"]).copy()
df["review_text"] = df["review_text"].astype(str)
df["score"] = df["score"].astype(int)

X = df["review_text"]
y = df["score"]

# Stratified split er fint for klassefordeling (1-10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 2) Build pipeline: TF-IDF -> kNN
# ----------------------------
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=2,                # ignorer super-sjeldne tokens
        max_df=0.95,
        sublinear_tf=True
    )),
    ("knn", KNeighborsClassifier())
])

# ----------------------------
# 3) Hyperparameter search
# ----------------------------
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 11, 15, 21],
    "knn__weights": ["uniform", "distance"],
    # cosine fungerer ofte bra for tekst (i stedet for euclidean)
    "knn__metric": ["cosine", "euclidean"],
}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="f1_macro",   # macro F1 gir mer rettferdig score ved ubalanse
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("\nBest params:", grid.best_params_)
print("Best CV f1_macro:", grid.best_score_)

# ----------------------------
# 4) Evaluate on test set
# ----------------------------
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

# MAE: behandler score som tall (1-10) og måler hvor mange score-poeng feil i snitt
mae = mean_absolute_error(y_test.astype(float), y_pred.astype(float))

print("\n--- Test metrics ---")
print("Accuracy:", round(acc, 4))
print("Macro F1:", round(f1m, 4))
print("MAE (avg score error):", round(mae, 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion matrix (rows=true, cols=pred):")
labels = list(range(1, 11))
cm = confusion_matrix(y_test, y_pred, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels))

# ----------------------------
# 5) (Optional) Save predictions to CSV
# ----------------------------
out = pd.DataFrame({
    "review_text": X_test.values,
    "true_score": y_test.values,
    "pred_score": y_pred
})
out.to_csv("knn_predictions.csv", index=False)
print("\nSaved predictions to: knn_predictions.csv")
