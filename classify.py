import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def classify_tweet(tweet):
    # Load the tokenizer
    with open('./trained_models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load max_length
    with open('./trained_models/max_length.txt', 'r') as f:
        max_length = int(f.read())

    # Tokenize and pad the input tweet
    sequences = tokenizer.texts_to_sequences([tweet])
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_length)

    # Load the trained model
    model = tf.keras.models.load_model('./trained_models/gru_tweet_classifier.h5')

    # Make prediction
    prediction = model.predict(padded_sequences)
    prediction = (prediction > 0.1).astype(int).flatten()

    # Output result
    if prediction[0] == 1:
        print("The tweet is classified as: Harmful")
    else:
        print("The tweet is classified as: Normal")

# Example usage
if __name__ == "__main__":
    user_tweet = input("Enter a tweet to classify: ")
    classify_tweet(user_tweet)
