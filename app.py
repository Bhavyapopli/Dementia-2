import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import nltk
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

# Streamlit Page Config
st.set_page_config(page_title="🧠 Dementia Detection Inference", layout="centered")
st.title("🧠 Dementia Detection - Inference App")

# Load Glove Word2Vec model and scaler on app start
@st.cache_resource(show_spinner=True)
def load_resources():
    glove_model = KeyedVectors.load_word2vec_format("glove.6B.100d.word2vec.txt", binary=False)
    scaler = joblib.load("models/scaler.pkl")
    return glove_model, scaler

glove_model, scaler = load_resources()

stop_words = set(stopwords.words("english"))

# Function to extract text from .cha
def extract_text_from_cha(file_content):
    extracted_text = []
    for line in file_content.decode("utf-8").split("\n"):
        line = line.strip()
        if line.startswith(("*PAR:", "*CHI:", "*SPE:")):
            line = re.sub(r"\[.*?\]", "", line)
            line = re.sub(r"[^a-zA-Z\s:]", "", line)
            if ":" in line:
                extracted_text.append(line.split(":", 1)[1].strip().lower())
    return " ".join(extracted_text) if extracted_text else "empty_transcription"

# Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens) if tokens else "meaningful_word"

# Convert sentence to vector
def sentence_vector(text, model, vector_size=100):
    tokens = text.split()
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# Load available models from directory
def get_available_models():
    model_files = [f for f in os.listdir("saved_models") if f.endswith(".pkl")]
    return model_files

# File Upload
cha_file = st.file_uploader("Upload a .cha Transcript File", type=["cha"])

# Model Selection
model_files = get_available_models()
selected_model_name = st.selectbox("Select a Trained Model", options=model_files)

# Run inference if both provided
if cha_file and selected_model_name:
    # Load model
    model_path = os.path.join("saved_models", selected_model_name)
    model = joblib.load(model_path)

    # Process text
    raw_text = extract_text_from_cha(cha_file.read())
    processed_text = preprocess_text(raw_text)

    if processed_text == "meaningful_word" or processed_text == "":
        st.error("No valid transcription data found in file.")
    else:
        vector = sentence_vector(processed_text, glove_model)
        vector_scaled = scaler.transform([vector])

        # Inference timing
        start_time = time.time()
        prediction = model.predict(vector_scaled)
        probability = model.predict_proba(vector_scaled)[0][1]
        end_time = time.time()
        inference_time = end_time - start_time

        label = "Control (Healthy)" if prediction[0] == 0 else "🔴 Dementia"

        st.success(f"**Prediction:** {label}")
        st.write(f"**Dementia Probability:** {probability:.4f}")
        st.write(f"**Inference Time:** {inference_time:.4f} seconds")

else:
    st.info("Upload a .cha file and select a trained model to run inference.")
