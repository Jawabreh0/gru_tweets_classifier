import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
import pickle

# Load the datasets
normal_training = pd.read_csv('dataset/training_dataset/normal.csv')
harmful_training = pd.read_csv('dataset/training_dataset/harmful.csv')

# Combine datasets and create labels
normal_training['label'] = 0
harmful_training['label'] = 1
training_data = pd.concat([normal_training, harmful_training])

# Shuffle the training data
training_data = training_data.sample(frac=1).reset_index(drop=True)

# Ensure all tweets are strings and fill missing values
training_data['tweet'] = training_data['tweet'].astype(str).fillna('')

# Prepare the data for training
tweets = training_data['tweet'].values
labels = training_data['label'].values

# Tokenization and padding
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(tweets)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(tweets)
padded_sequences = pad_sequences(sequences, padding='post')

# Save the tokenizer for later use
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Define model parameters
vocab_size = len(word_index) + 1
embedding_dim = 100
max_length = padded_sequences.shape[1]

# Save max_length for later use
with open('max_length.txt', 'w') as f:
    f.write(str(max_length))

# Build the GRU model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    GRU(128, return_sequences=True),
    Dropout(0.2),
    GRU(128),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=50, batch_size=32)

# Save the model
model.save('gru_tweet_classifier.h5')
