import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
@st.cache_data
def get_word_indexes():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

word_index, reverse_word_index = get_word_indexes()

# Build absolute path to locate model file reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'simple_rnn_imdb.h5')

# Cache model loading to avoid reloading on every interaction
@st.cache_resource
def load_keras_model(path):
    return load_model(path)

model = load_keras_model(model_path)


# Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Streamlit App UI
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip():
        preprocessed_input = preprocess_text(user_input)
        
        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display results
        st.write(f'**Sentiment:** {sentiment}')
        st.write(f'**Prediction Score:** {prediction[0][0]:.4f}')
    else:
        st.warning('Please enter a valid movie review before classifying.')
