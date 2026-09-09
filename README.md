#  Chatbot for FAQs

##  Project Overview

This project is an **FAQ Chatbot** developed as part of **CodeAlpha Task 2**.
The chatbot uses **Natural Language Processing (NLP)** to understand user questions and find the most relevant answer from a collection of Frequently Asked Questions (FAQs).

The project preprocesses text, converts questions into numerical vectors using **TF-IDF**, and uses **Cosine Similarity** to find the FAQ that is most similar to the user's question.

##  Objectives

* Collect and prepare FAQ questions and answers.
* Clean and preprocess text using NLP techniques.
* Convert text into numerical features using TF-IDF.
* Match user questions with the most similar FAQ.
* Display the best matching answer as a chatbot response.
* Provide a simple user interface for interaction.

##  Technologies Used

* **Python**
* **NLTK**
* **Scikit-learn**
* **Pandas**
* **Streamlit**
* **TF-IDF Vectorization**
* **Cosine Similarity**

##  How It Works

1. FAQ questions and answers are collected in a dataset.
2. The text is cleaned and preprocessed using NLP techniques.
3. Stopwords and unnecessary characters are removed.
4. The FAQ questions are converted into TF-IDF vectors.
5. The user's question is also converted into a TF-IDF vector.
6. Cosine Similarity is calculated between the user's question and FAQ questions.
7. The FAQ with the highest similarity is selected.
8. The corresponding answer is displayed to the user.

##  Project Structure

```text
FAQ_chatbot/
│
├── app.py
├── chatbot_ui.py
├── clean_final.py
├── final_dataset.csv
├── requirements.txt
├── test_clean.py
└── test_final.py
```

##  How to Run

### 1. Clone the repository

```bash
git clone https://github.com/shah7205khushi/CodeAlpha_Chatbot-for-FAQs.git
```

### 2. Open the project folder

```bash
cd CodeAlpha_Chatbot-for-FAQs
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the chatbot

```bash
streamlit run app.py
```

The chatbot will open in your browser.

## 💬 Example

**User:**
`How can I reset my password?`

**Chatbot:**
The chatbot searches the FAQ dataset and returns the answer corresponding to the most similar question.

##  Testing

The project also contains test files:

* `test_clean.py`
* `test_final.py`

These files are used to test the data cleaning and final chatbot functionality.

##  Developed By

**Khushi Shah**

### CodeAlpha Internship — Task 2

**Chatbot for FAQs**
