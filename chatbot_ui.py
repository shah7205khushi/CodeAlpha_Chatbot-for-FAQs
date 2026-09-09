import streamlit as st
import pandas as pd
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)


# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("final_dataset.csv")

stop_words = set(stopwords.words("english"))


# =====================================
# TEXT PREPROCESSING
# =====================================

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


# =====================================
# PREPROCESS FAQ QUESTIONS
# =====================================

df["clean_input"] = df["input"].apply(
    preprocess_text
)


# =====================================
# TF-IDF VECTORIZATION
# =====================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

faq_vectors = vectorizer.fit_transform(
    df["clean_input"]
)


# =====================================
# VALID FAQ CHECK
# =====================================

def is_valid_faq(question):

    question = str(question).strip()

    if not question:
        return False

    if len(question.split()) <= 1:
        return False

    if question.startswith('"') and not question.endswith('"'):
        return False

    return True


# =====================================
# FIND BEST ANSWER
# =====================================

def find_best_answer(user_question):

    # Greeting support
    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "good morning",
        "good evening"
    ]

    if user_question.lower().strip() in greetings:
        return (
            "Hello! How can I help you today?",
            1.0
        )

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

    top_indices = similarities.argsort()[-10:][::-1]

    best_index = None
    best_score = 0

    for index in top_indices:

        question = df.iloc[index]["input"]

        if not is_valid_faq(question):
            continue

        score = similarities[index]

        if score > best_score:
            best_score = score
            best_index = index

    threshold = 0.70

    if best_index is None or best_score < threshold:

        return (
            "Sorry, I couldn't find a relevant answer to your question.",
            best_score
        )

    answer = df.iloc[best_index]["target"]

    return (
        answer,
        best_score
    )


# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main-title{
    text-align:center;
    color:#1f77b4;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:gray;
    margin-bottom:25px;
}

.result-box{
    padding:20px;
    border-radius:10px;
    background:#f0f8ff;
    border-left:5px solid #1f77b4;
    margin-top:15px;
}

.score-box{
    background:#eef7ff;
    padding:10px;
    border-radius:8px;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# UI
# =====================================

st.markdown(
    '<div class="main-title">🤖 FAQ Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Ask any FAQ question and get an answer instantly</div>',
    unsafe_allow_html=True
)

user_question = st.text_input(
    "Enter your question:"
)

if st.button("Ask Question"):

    if user_question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        answer, score = find_best_answer(
            user_question
        )

        st.markdown(
            f"""
            <div class="result-box">
            <h3>Answer</h3>
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="score-box">
            Similarity Score: {score:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "Built using NLP, TF-IDF Vectorization and Cosine Similarity"
)