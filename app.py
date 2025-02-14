import streamlit as st
import pickle
import numpy as np
import pandas as pd
import speech_recognition as sr
import pyttsx3
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from transformers import pipeline

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="AI Spam Detector", layout="wide", page_icon="📧")

# ✅ Load Model and Vectorizer
@st.cache_resource
def load_model():
    with open("spam_classifier_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_vectorizer():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
vectorizer = load_vectorizer()

# ✅ Sidebar UI
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/SpamEmail.svg/800px-SpamEmail.svg.png", width=150)
    st.title("📧 AI Spam Detector")
    st.info("Enter text, upload a file, or use speech input to detect spam emails.")

# ✅ Main UI
st.markdown("<h2 style='text-align: center;'>🚀 AI-Powered Spam Mail Detector</h2>", unsafe_allow_html=True)
email_text = st.text_area("📩 Enter email text:", height=150)

# ✅ File Upload
uploaded_file = st.file_uploader("📂 Or upload a `.txt` file", type=["txt"])
if uploaded_file is not None:
    email_text = uploaded_file.getvalue().decode("utf-8")

# ✅ Speech Input
if st.button("🎤 Record Email Using Microphone"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎧 Speak now...")
        try:
            audio = recognizer.listen(source, timeout=10)
            email_text = recognizer.recognize_google(audio)
            st.success(f"✅ Captured Text: {email_text}")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# ✅ Spam Detection
if st.button("🔍 Detect Spam"):
    if email_text.strip():
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]
        confidence = model.predict_proba(email_vector)[0][1]
        
        result_msg = "🚨 Spam Detected!" if prediction == 1 else "✅ Not Spam!"
        st.subheader(f"🔹 {result_msg} (Confidence: {confidence:.2%})")

        # ✅ Spam Score Analysis
        spam_score = round(confidence * 100, 2)
        st.info(f"📊 **Spam Score:** {spam_score}%")

        # ✅ Keyword Extraction
        word_counts = Counter(email_text.split())
        top_keywords = ", ".join([word for word, _ in word_counts.most_common(5)])
        st.info(f"🔑 **Top Keywords:** {top_keywords}")

        # ✅ Email Summarization
        summarizer = pipeline("summarization")
        summary = summarizer(email_text, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
        st.success(f"🔄 **Summary:** {summary}")
    else:
        st.warning("⚠️ Please enter text or upload a file.")

# ✅ Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Developed with ❤️ by Ujwal Reddy</div>", unsafe_allow_html=True)
