import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# ---------------------------
# Load NLTK stopwords safely
# ---------------------------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# ---------------------------
# Load model & vectorizer safely
# ---------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

model, tfidf = load_model()

# ---------------------------
# Text preprocessing
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ---------------------------
# UI
# ---------------------------
st.title("📰 Fake News Detector")

user_input = st.text_area("Paste a news article here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text before classifying.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ This news is likely REAL.")
        else:
            st.error("❌ This news is likely FAKE.")
