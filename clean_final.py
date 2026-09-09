import pandas as pd
import re


# =====================================================
# 1. LOAD CLEAN DATASET
# =====================================================

df = pd.read_csv("clean_dataset.csv")

print("Original clean dataset:", len(df))


# =====================================================
# 2. REMOVE EMPTY QUESTIONS / ANSWERS
# =====================================================

df = df.dropna(subset=["input", "target"])

df = df[
    (df["input"].str.strip() != "") &
    (df["target"].str.strip() != "")
]


# =====================================================
# 3. CLEAN EXTRA WHITESPACE
# =====================================================

df["input"] = (
    df["input"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df["target"] = (
    df["target"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# =====================================================
# 4. REMOVE CLEARLY MALFORMED QUESTIONS
# =====================================================

def is_malformed(question):

    q = question.strip()

    # Too short to be a meaningful FAQ
    if len(q.split()) <= 1:
        return True

    # Questions ending with an obvious incomplete phrase
    incomplete_endings = [
        "what",
        "who",
        "why",
        "how",
        "when",
        "where",
        "which",
        "the",
        "a",
        "an",
        "to",
        "of",
        "in",
        "for",
        "with"
    ]

    last_word = q.lower().strip(" ?.!\"'")

    if last_word in incomplete_endings and len(q.split()) <= 3:
        return True

    # Obvious broken fragments
    broken_patterns = [
        r'^"who\s*$',
        r'^"why\s*$',
        r'^"what\s*$',
        r'^"how\s*$',
        r'^"when\s*$',
        r'^"where\s*$',
        r'^"which\s*$',
        r'^"whatlanguage\s*$'
    ]

    for pattern in broken_patterns:
        if re.match(pattern, q.lower()):
            return True

    return False


df = df[
    ~df["input"].apply(is_malformed)
].copy()


# =====================================================
# 5. REMOVE DUPLICATE QUESTIONS
# =====================================================

df = df.drop_duplicates(
    subset=["input"],
    keep="first"
)


# =====================================================
# 6. RESET INDEX
# =====================================================

df = df.reset_index(drop=True)


# =====================================================
# 7. SAVE FINAL DATASET
# =====================================================

df.to_csv(
    "final_dataset.csv",
    index=False
)


# =====================================================
# 8. DISPLAY RESULTS
# =====================================================

print("\nFinal dataset size:", len(df))

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFirst 10 questions:")

print(
    df[
        ["input", "target"]
    ].head(10).to_string(index=False)
)

print("\nFinal dataset saved as final_dataset.csv")