import re
import math
import pandas as pd

# --- Load ---
path_in = "reviews_clean.csv"
df = pd.read_csv(path_in)

# --- Keyword lists (edit freely) ---
# Higher weight = more "useful" for debugging/support
PHRASES = {
    # Critical / high-priority issues
    r"\bhack(ed|ing)?\b": 6,
    r"\bbann?ed\b": 6,
    r"\bspam\b": 5,
    r"\b(can'?t|cannot)\s+(open|use|login|sign in)\b": 5,
    r"\b(crash|crashes)\b": 5,
    r"\b(close(s)?\s+automatically|force\s+close)\b": 5,
    r"\bnot\s+working\b": 4,
    r"\boffline\b": 3,
    r"\burgent\b": 3,

    # Feature / component issues
    r"\bwhatsapp\s+web\b": 4,
    r"\bcamera\b": 3,
    r"\bfocus\b": 2,
    r"\binstall\b": 2,
    r"\bverification\b": 2,
    r"\botp\b": 2,
    r"\baccount\b": 2,

    # Useful context / entities
    r"\b(feature|privacy|chat\s+lock)\b": 2,
    r"\b(redmi|oppo|samsung|xiaomi|iphone|android)\b": 2,  # device/brand hints
}

# Words that tend to indicate generic praise (low diagnostic value)
GENERIC_POSITIVE = re.compile(r"\b(nice|good|great|awesome|excellent|love|best|wonderful|thnx|thank(s)?)\b", re.I)

def tokenize(text: str):
    return re.findall(r"[a-z0-9']+", (text or "").lower())

def repetition_penalty(tokens):
    """Return penalty points (0..8) for repetitive/noisy text."""
    if not tokens:
        return 0
    total = len(tokens)
    uniq = len(set(tokens))
    unique_ratio = uniq / total
    # Most frequent token share
    from collections import Counter
    c = Counter(tokens)
    top_share = c.most_common(1)[0][1] / total

    penalty = 0
    # Very repetitive or "word-salad" style
    if unique_ratio < 0.35:
        penalty += 4
    if top_share > 0.25:
        penalty += 4
    return min(penalty, 8)

def length_points(n_words: int) -> int:
    """
    Small boost for having enough text, capped to avoid rewarding spam.
    Returns 0..4
    """
    if n_words < 4:
        return 0
    if n_words < 10:
        return 1
    if n_words < 25:
        return 2
    if n_words < 60:
        return 3
    return 4

def keyword_points(text: str) -> int:
    """Sum keyword/phrase weights, capped to keep 1–10 mapping stable."""
    s = 0
    for pat, w in PHRASES.items():
        if re.search(pat, text or "", flags=re.I):
            s += w
    return min(s, 14)  # cap

def generic_positive_penalty(text: str) -> int:
    """
    If it's mostly generic praise and no issue keywords, reduce usefulness a bit.
    Returns 0..2
    """
    t = text or ""
    has_positive = bool(GENERIC_POSITIVE.search(t))
    has_issue_kw = any(re.search(p, t, flags=re.I) for p in PHRASES.keys())
    if has_positive and not has_issue_kw:
        return 2
    return 0

def raw_usefulness_score(text: str) -> float:
    tokens = tokenize(text)
    n_words = len(tokens)

    lp = length_points(n_words)               # 0..4
    kp = keyword_points(text)                 # 0..14
    rp = repetition_penalty(tokens)           # 0..8
    gp = generic_positive_penalty(text)       # 0..2

    raw = (kp * 1.0) + (lp * 1.2) - (rp * 1.3) - (gp * 1.0)
    return raw

def to_1_10(raw: float) -> int:
    """
    Map raw score to 1..10 with a smooth-ish scaling.
    Adjust thresholds here if you want stricter/looser scoring.
    """
    # Shift & scale then clamp
    # Typical raw values end up ~[-5, 18]
    scaled = 1 + (raw + 4) * (9 / 22)  # maps raw=-4 -> ~1, raw=18 -> ~10
    return int(max(1, min(10, round(scaled))))

# --- Compute score ---
df["score"] = df["review_text"].astype(str).apply(lambda t: to_1_10(raw_usefulness_score(t)))

# --- Insert score right after 'helpful' column ---
if "helpful" in df.columns:
    cols = list(df.columns)
    cols.remove("score")
    helpful_idx = cols.index("helpful")
    cols.insert(helpful_idx + 1, "score")
    df = df[cols]

# --- Save as new CSV ---
path_out = "Datacleaned_reviews.csv"
df.to_csv(path_out, index=False)
print("New file saved to:", path_out)
