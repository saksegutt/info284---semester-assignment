import pandas as pd
import re

# Les fil
df = pd.read_csv("reviews.csv")

def is_noise(text):
    if not isinstance(text, str):
        return True

    text = text.strip()

    # 1. For kort
    if len(text) < 15:
        return True

    # 2. Bare tall
    if text.isdigit():
        return True

    # 3. For mye repetisjon av samme tegn
    most_common_char_ratio = max(text.count(c) for c in set(text)) / len(text)
    if most_common_char_ratio > 0.7:
        return True

    # 4. Random string (veldig få vokaler)
    vowels = len(re.findall(r"[aeiouAEIOU]", text))
    if vowels / len(text) < 0.1:
        return True

    return False

# Fjern støy
df_clean = df[~df["review_text"].apply(is_noise)]

# Fjern duplikater
df_clean = df_clean.drop_duplicates(subset=["review_text"])

# Lagre ny fil
df_clean.to_csv("reviews_clean.csv", index=False)