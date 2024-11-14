import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = load_model("sent_dep_model.h5")

# Load the saved tokenizer used during training
with open("tokenizer.pkl", "rb") as tk:
    tokenizer = pickle.load(tk)

#  Define the function to preprocess the user text input
def preprocess_text(text):
    # Tokenize the text
    tokens = tokenizer.texts_to_sequences([text])

    # Pad the sequences to a fixed length
    padded_tokens = pad_sequences(tokens, maxlen = 100)
    return padded_tokens[0]

# Create the title of the app
st.title("Sentiment Analysis App")

# Create a text input widget for the user input
user_input = st.text_area("Enter text for sentiment analysis", " ")

# Create a button to trigger the sentiment analysis
if st.button("Predict Sentiment"):
    # Preprocess the user input
    processed_input = preprocess_text(user_input)

    # Make prediction using the loaded model
    prediction = model.predict(np.array([processed_input]))
    # Write the prediction to the screen
    st.write(prediction)

    # Create the sentiment
    sentiment = "Negative" if prediction[0][0] > 0.5 else "Positive"

    # Display the sentiment
    st.write(f" ### Sentiment: {sentiment}")
