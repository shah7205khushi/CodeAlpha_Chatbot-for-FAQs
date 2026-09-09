import pandas as pd
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. LOAD FINAL DATASET
# =====================================================

df = pd.read_csv("final_dataset.csv")

print("Final dataset size:", len(df))


# =====================================================
# 2. NLTK SETUP
# =====================================================

stop_words = set(stopwords.words("english"))


# =====================================================
# 3. PREPROCESSING
# =====================================================

def preprocess_text(text):

    text = str(text).lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# =====================================================
# 4. PREPROCESS FAQ QUESTIONS
# =====================================================

df["clean_input"] = df["input"].apply(
    preprocess_text
)


# =====================================================
# 5. TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

faq_vectors = vectorizer.fit_transform(
    df["clean_input"]
)

print("TF-IDF shape:", faq_vectors.shape)


# =====================================================
# 6. FIND BEST MATCH
# =====================================================

def find_best_match(user_question):

    clean_question = preprocess_text(
        user_question
    )

    user_vector = vectorizer.transform(
        [clean_question]
    )

    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    # Get top 10 candidates
    top_indices = similarities.argsort()[-10:][::-1]

    return top_indices, similarities


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
# 8. RUN TEST
# =====================================================

for question in test_questions:

    top_indices, similarities = find_best_match(
        question
    )

    print("\n")
    print("=" * 60)

    print("User Question:")
    print(question)

    print("\nTop 10 Matches:")

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        print("\n", rank)
        print("Question:", df.iloc[index]["input"])
        print("Score:", similarities[index])
        print("Answer:", df.iloc[index]["target"])