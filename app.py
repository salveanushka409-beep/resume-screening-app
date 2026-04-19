import streamlit as st
import pickle

st.title("AI Resume Screening System")

# Load model & vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except:
    st.warning("⚠️ Model files not found. Please add model.pkl and vectorizer.pkl")
    model = None
    vectorizer = None

# Input box
resume_text = st.text_area("Paste Resume Text Here")

# File upload
uploaded_file = st.file_uploader("Upload Resume (txt)", type=["txt"])

if uploaded_file:
    resume_text = uploaded_file.read().decode("utf-8")

# Predict button
if st.button("Predict"):
    if model and vectorizer:
        if resume_text.strip() != "":
            transformed = vectorizer.transform([resume_text])
            prediction = model.predict(transformed)

            st.success(f"Predicted Category: {prediction[0]}")
        else:
            st.warning("Please enter resume text")
    else:
        st.error("Model not loaded")