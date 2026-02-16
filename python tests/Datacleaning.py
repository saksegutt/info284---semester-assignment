import pandas as pd
import re

# =========================
# CONFIG
# =========================
INPUT_CSV = "reviews.csv"
OUTPUT_CSV = "reviews_cleaned.csv"

TEXT_COL = "review_text"   # <-- endre hvis kolonnen heter noe annet
MIN_TEXT_LEN = 10          # fjern veldig korte reviews

# Hvor mye du vil printe:
PRINT_FULL_DATASET = False  # True = skriver hele cleaned datasettet til terminal (kan bli veldig mye)
PRINT_ROWS = 20             # antall rader å vise hvis PRINT_FULL_DATASET=False

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


# =========================
# TEXT CLEANING FUNCTION
# =========================
def clean_text(text: str) -> str:
    """Basic text normalization for analysis/modeling."""
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # remove HTML tags (if any)
    text = re.sub(r"<.*?>", " ", text)

    # keep letters (incl. æøå) and spaces
    text = re.sub(r"[^a-zæøå\s]", " ", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# MAIN
# =========================
def main():
    # ---------- Load ----------
    df = pd.read_csv(INPUT_CSV)
    print("\n=== Loaded dataset ===")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("Columns:", list(df.columns))

    if TEXT_COL not in df.columns:
        raise ValueError(
            f"Fant ikke tekstkolonnen '{TEXT_COL}'. "
            f"Tilgjengelige kolonner: {list(df.columns)}"
        )

    # ---------- Basic EDA ----------
    print("\n=== EDA (before cleaning) ===")
    missing_text = df[TEXT_COL].isna().sum()
