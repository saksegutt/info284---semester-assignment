import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer #fjern den her trur ej, trenge bere i trym.py

dataset = pd.read_csv("Main/reviews.csv")

# Fjern rating 3 (nøytral)
dataset = dataset[dataset['rating'] != 3]

# Lag binær målvariabel (1-2 = negativ, 4-5 = positiv)
dataset['sentiment'] = dataset['rating'].apply(lambda x: 1 if x >= 4 else 0)

#del opp i trenings- og testsett
X_train, X_test, y_train, y_test = train_test_split(
    dataset['review_text'], dataset['sentiment'], test_size=0.2, random_state=42)




#chattern-----------------# FJERN NÅR VI LEGGE DEN INN I MAINMASTER, 
# må bere ha her for å funke utenfor mainmaster, trur ej
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
#chattern-----------------




#Tren en Random Forest-klassifikator
randomforest=RandomForestClassifier(n_estimators=100, random_state=42)
randomforest.fit(X_train, y_train)


#prediction her
y_pred = randomforest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)
