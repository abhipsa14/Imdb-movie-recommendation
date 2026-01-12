## step 1 import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

## load the IMDB dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

# load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# step 2: helper function
# function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# step 3: prediction function
def predict_sentiment(text):
    processed_text=preprocess_text(text)
    prediction=model.predict(processed_text)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment, prediction[0][0]

## streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")
## user input 
user_input=st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)

    # make prediction
    prediction = predict_sentiment(preprocessed_input)

    # display the result
    st.write(f"Predicted Sentiment: {prediction[0]}")
    st.write(f"Confidence Score: {prediction[1]:.4f}")
else:
    st.write("Please enter a movie review and click 'Classify' to see the sentiment prediction or please enter a movie review")

