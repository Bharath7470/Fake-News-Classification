import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords (only once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Preprocess input
def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# UI layout
st.title("📰 Fake News Detector")
user_input = st.text_area("Paste a news article here:")

if st.button("Classify"):
    cleaned = clean_text(user_input)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.success("✅ This news is likely REAL.")
    else:
        st.error("❌ This news is likely FAKE.")