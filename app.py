# app.py
import streamlit as st
import numpy as np
import joblib
import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
import os

# === Setup ===
nltk.download('stopwords')

# === Rebuild split model file ===
def reconstruct_file(base_name):
    if os.path.exists(base_name):
        return  # Already reconstructed

    with open(base_name, "wb") as out_file:
        i = 0
        while True:
            part_file = f"{base_name}.part{i}"
            if not os.path.exists(part_file):
                break
            with open(part_file, "rb") as pf:
                out_file.write(pf.read())
            i += 1

# === Reconstruct files ===
reconstruct_file("fake_news_model.pkl")
reconstruct_file("tfidf_vectorizer.pkl")

# === Load model and vectorizer ===
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# === Text Cleaning ===
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# === Streamlit App UI ===
st.set_page_config(page_title="Fake News Detector")
st.title("Fake News Detector")
st.markdown("Type or paste a news **headline or short article** below to check if it's **real or fake**.")

user_input = st.text_area("🖊️ Enter News Text Here")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        polarity = TextBlob(cleaned).sentiment.polarity

        input_vec = tfidf.transform([cleaned])
        full_input = np.hstack([input_vec.toarray(), [[polarity]]])

        prediction = model.predict(full_input)[0]
        confidence = round(np.max(model.predict_proba(full_input)) * 100, 2)

        st.subheader("🧠 Prediction Result:")
        if prediction == 0:
            st.success(f"✅ This news is **REAL** ({confidence}% confidence)")
        else:
            st.error(f"❌ This news is **FAKE** ({confidence}% confidence)")
