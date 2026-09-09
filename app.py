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
# 3. TEXT PREPROCESSING
# =====================================================

def preprocess_text(text):

    text = str(text).lower()

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

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

faq_vectors = vectorizer.fit_transform(
    df["clean_input"]
)

print("\nTF-IDF vectorization completed.")
print("Shape of TF-IDF matrix:", faq_vectors.shape)


# =====================================================
# 6. CHECK INCOMPLETE FAQ
# =====================================================

def is_valid_faq(question):

    question = str(question).strip()

    # Reject empty questions
    if not question:
        return False

    # Reject very short questions
    if len(question.split()) <= 1:
        return False

    # Reject questions with an unmatched opening quote
    if question.startswith('"') and not question.endswith('"'):
        return False

    # Reject obvious incomplete fragments
    incomplete_questions = [
        '"who',
        '"what',
        '"why',
        '"how',
        '"when',
        '"where',
        '"which'
    ]

    question_lower = question.lower().strip()

    for phrase in incomplete_questions:
        if question_lower == phrase:
            return False

    return True


# =====================================================
# 7. FIND BEST FAQ ANSWER
# =====================================================

def find_best_answer(user_question):

    # Preprocess user question
    clean_question = preprocess_text(
        user_question
    )

    # Convert user question into TF-IDF vector
    user_vector = vectorizer.transform(
        [clean_question]
    )

    # Calculate cosine similarity
    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    # Get top 10 candidates
    top_indices = similarities.argsort()[-10:][::-1]

    # Find best valid FAQ
    best_index = None
    best_score = 0

    for index in top_indices:

        question = df.iloc[index]["input"]

        # Skip malformed FAQ questions
        if not is_valid_faq(question):
            continue

        score = similarities[index]

        if score > best_score:
            best_score = score
            best_index = index

    # Similarity threshold
    threshold = 0.70

    # No relevant FAQ found
    if best_index is None or best_score < threshold:

        return (
            "Sorry, I couldn't find a relevant answer to your question.",
            best_score,
            None
        )

    # Get answer
    answer = df.iloc[best_index]["target"]

    return (
        answer,
        best_score,
        df.iloc[best_index]["input"]
    )


# =====================================================
# 8. TEST QUESTIONS
# =====================================================

test_questions = [

    "Who invented the Hebbian learning rule?",

    "What is the first neural network called?",

    "What is deep learning?",

    "What is the weather today?"
]


# =====================================================
# 9. TEST CHATBOT
# =====================================================

for question in test_questions:

    answer, score, matched_question = find_best_answer(
        question
    )

    print("\n")
    print("=" * 60)

    print("User Question:")
    print(question)

    print("\nMatched FAQ:")
    print(matched_question)

    print("\nSimilarity Score:")
    print(score)

    print("\nChatbot Answer:")
    print(answer)