import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and files
model = load_model("model/feedback_model.h5")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Title
st.title("Student Feedback Analyzer")

# Input
feedback = st.text_area("Enter Student Feedback")

# Button
if st.button("Analyze"):

    if feedback.strip() == "":
        st.warning("Please enter feedback.")
    else:

        # Convert text to sequence
        seq = tokenizer.texts_to_sequences([feedback])

        # Padding
        padded = pad_sequences(seq, maxlen=100)

        # Prediction
        pred = model.predict(padded)

        # Sentiment
        result = encoder.inverse_transform(
            [np.argmax(pred)]
        )[0]

        # Confidence
        confidence = np.max(pred) * 100

        st.success(f"Sentiment: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Probabilities
        st.subheader("Prediction Probabilities")

        for label, prob in zip(encoder.classes_, pred[0]):
            st.write(f"{label}: {prob*100:.2f}%")