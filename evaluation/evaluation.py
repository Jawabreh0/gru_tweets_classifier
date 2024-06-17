import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the validation datasets
normal_validation = pd.read_csv('../dataset/testing_dataset/testing-normal.csv')
harmful_validation = pd.read_csv('../dataset/testing_dataset/testing-harmful.csv')

# Combine datasets and create labels
normal_validation['label'] = 0
harmful_validation['label'] = 1
validation_data = pd.concat([normal_validation, harmful_validation])

# Ensure all tweets are strings and fill missing values
validation_data['tweet'] = validation_data['tweet'].astype(str).fillna('')

# Prepare the data for validation
tweets = validation_data['tweet'].values
labels = validation_data['label'].values

# Load the tokenizer
with open('../trained_models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load max_length
with open('../trained_models/max_length.txt', 'r') as f:
    max_length = int(f.read())

# Tokenization and padding using the same tokenizer fitted on the training data
sequences = tokenizer.texts_to_sequences(tweets)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_length)

# Load the trained model
model = tf.keras.models.load_model('../trained_models/gru_tweet_classifier.h5')

# Make predictions
predictions = model.predict(padded_sequences)
predictions = (predictions > 0.5).astype(int).flatten()

# Calculate metrics
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
conf_matrix = confusion_matrix(labels, predictions)

# Print results
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Harmful'], yticklabels=['Normal', 'Harmful'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Extracting precision, recall, f1-score for visualization
report_dict = classification_report(labels, predictions, target_names=['normal', 'harmful'], output_dict=True)
categories = list(report_dict.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
precision = [report_dict[cat]['precision'] for cat in categories]
recall = [report_dict[cat]['recall'] for cat in categories]
f1_score = [report_dict[cat]['f1-score'] for cat in categories]

x = np.arange(len(categories))  # the label locations
width = 0.2  # the width of the bars

# Plotting precision, recall, f1-score
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by category and metric')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Function to label the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()
