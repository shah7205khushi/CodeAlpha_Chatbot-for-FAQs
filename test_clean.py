import pandas as pd
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. LOAD CLEAN DATASET
# =====================================================

df = pd.read_csv("clean_dataset.csv")

print("Clean dataset size:", len(df))


# =====================================================
# 2. NLTK SETUP
# =====================================================

stop_words = set(stopwords.words("english"))


# =====================================================
# 3. TEXT PREPROCESSING
# =====================================================

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Convert tokens back to text
    return " ".join(tokens)


# =====================================================
# 4. PREPROCESS FAQ QUESTIONS
# =====================================================

df["clean_input"] = df["input"].apply(
    preprocess_text
)


# =====================================================
# 5. TF-IDF VECTORIZATION
# =====================================================

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    df["clean_input"]
)

print("TF-IDF shape:", faq_vectors.shape)


# =====================================================
# 6. FIND BEST FAQ
# =====================================================

def find_best_answer(user_question):

    # Preprocess user question
    clean_question = preprocess_text(
        user_question
    )

    # Convert user question to TF-IDF
    user_vector = vectorizer.transform(
        [clean_question]
    )

    # Calculate cosine similarity
    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )

    # Get top 5 matches
    top_indices = similarities[0].argsort()[-5:][::-1]

    # Best match
    best_index = top_indices[0]

    best_score = similarities[0][best_index]

    return (
        best_index,
        best_score,
        top_indices,
        similarities
    )


# =====================================================
# 7. TEST QUESTIONS
# =====================================================

test_questions = [

    "Who invented the Hebbian learning rule?",

    "What is the first neural network called?",

    "What is deep learning?",

    "What is the weather today?"
]


# =====================================================
# 8. TEST CHATBOT
# =====================================================

for question in test_questions:

    (
        best_index,
        best_score,
        top_indices,
        similarities
    ) = find_best_answer(question)

    print("\n")
    print("=" * 60)

    print("\nUser Question:")
    print(question)

    print("\nBest Matched FAQ:")
    print(df.iloc[best_index]["input"])

    print("\nSimilarity Score:")
    print(best_score)

    print("\nAnswer:")
    print(df.iloc[best_index]["target"])

    # -------------------------------------------------
    # Top 5 Matches
    # -------------------------------------------------

    print("\nTop 5 Matches:")

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        print("\n", rank, ".")
        print("Question:", df.iloc[index]["input"])
        print("Score:", similarities[0][index])


# =====================================================
# 9. INSPECT DEEP LEARNING FAQs
# =====================================================

print("\n")
print("=" * 60)

print("\nDEEP LEARNING FAQs IN DATASET:")

deep_learning_questions = df[
    df["input"]
    .str.lower()
    .str.contains(
        "deep learning",
        na=False
    )
]


if len(deep_learning_questions) == 0:

    print("\nNo Deep Learning FAQs found.")

else:

    for _, row in deep_learning_questions.iterrows():

        print("\n----------------------------------------")

        print("\nQuestion:")
        print(row["input"])

        print("\nAnswer:")
        print(row["target"])


# =====================================================
# 10. CHECK SHORT / INCOMPLETE QUESTIONS
# =====================================================

print("\n")
print("=" * 60)

print("\nPOSSIBLY INCOMPLETE QUESTIONS:")

short_questions = df[
    df["clean_input"].str.split().str.len() <= 5
]

print(
    "Questions with 5 or fewer words:",
    len(short_questions)
)

print("\nExamples:")

print(
    short_questions[
        ["input", "target"]
    ].head(30).to_string(index=False)
)