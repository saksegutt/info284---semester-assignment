# Source - https://stackoverflow.com/a/46203314
# Posted by Anil_M, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-16, License - CC BY-SA 3.0

import csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read only the review_text column from CSV
texts = []
with open('reviews.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("review_text"):  # riktig kolonnenavn fra din fil
            texts.append(row["review_text"])

# Combine all reviews into one string
all_reviews = " ".join(texts)

# Generate word cloud from review text only
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(all_reviews)

# Show the word cloud
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
