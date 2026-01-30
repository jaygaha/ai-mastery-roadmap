"""
Simple RNN for Text Classification (Sentiment Analysis)

This script demonstrates how to build a Simple RNN for classifying text
sentiment (positive/negative). It covers the complete NLP pipeline from
raw text to trained model predictions.

Pipeline:
    1. Tokenization: Convert words to integer IDs
    2. Padding: Make all sequences the same length
    3. Embedding: Transform IDs into dense vectors
    4. SimpleRNN: Process the sequence and extract features
    5. Dense (sigmoid): Output probability of positive sentiment

Key concepts:
    - Why we need an Embedding layer for text
    - How padding handles variable-length inputs
    - Binary classification with sigmoid activation
    - Handling small datasets (overfitting considerations)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# =============================================================================
# 1. Prepare Text Data
# =============================================================================
# A small synthetic dataset of movie reviews with sentiments.
# In practice, you'd use larger datasets like IMDb (50,000 reviews).

reviews = [
    "This movie was fantastic and I loved it!",
    "I really enjoyed the plot and the acting was superb.",
    "A wonderful film, highly recommend.",
    "Absolutely brilliant, a must-watch.",
    "Terrible movie, very boring and a waste of time.",
    "I hated every minute of it, utterly dreadful.",
    "Not good, quite disappointing.",
    "Could have been better, somewhat dull."
]
sentiments = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 = positive, 0 = negative

# Add more samples to improve train/test split stability
# With very few samples, even a single misclassification is significant.
reviews += [
    "Great story and characters.",
    "I fell asleep, it was so boring.",
    "Best movie of the year!",
    "Awful, do not watch."
]
sentiments += [1, 0, 1, 0]

print(f"Dataset size: {len(reviews)} reviews")
print(f"Positive: {sum(sentiments)}, Negative: {len(sentiments) - sum(sentiments)}")


# =============================================================================
# 2. Tokenize and Vectorize Text Data
# =============================================================================
# The tokenizer learns a vocabulary and converts words to integer IDs.
# - num_words=1000: Keep only the 1000 most frequent words
# - oov_token="<unk>": Replace rare/unknown words with this token

tokenizer = Tokenizer(num_words=1000, oov_token="<unk>")
tokenizer.fit_on_texts(reviews)

# Convert each review to a sequence of word IDs
sequences = tokenizer.texts_to_sequences(reviews)

# Pad sequences so all have the same length
# Shorter sequences get zeros added; longer ones get truncated
maxlen = max(len(s) for s in sequences)
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

# Convert labels to numpy array for sklearn compatibility
labels = np.array(sentiments)

print(f"Vocabulary size: {len(tokenizer.word_index)} unique words")
print(f"Max sequence length: {maxlen}")


# =============================================================================
# 3. Split Data into Training and Testing Sets
# =============================================================================
# random_state=42 ensures reproducible splits
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


# =============================================================================
# 4. Build the Simple RNN Model for Text Classification
# =============================================================================
# vocab_size + 1 accounts for the padding token (index 0) and OOV token
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16  # Each word becomes a 16-dimensional vector

model = Sequential([
    # Embedding layer: Converts word IDs to dense vectors
    # Input: (batch_size, sequence_length) of integers
    # Output: (batch_size, sequence_length, embedding_dim)
    # Similar words will learn similar embeddings during training
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    
    # SimpleRNN: Processes the sequence one word at a time
    # Returns only the final hidden state by default (return_sequences=False)
    # This single vector captures information from the entire sequence
    SimpleRNN(units=32),
    
    # Dense with sigmoid: Outputs probability of positive sentiment
    # Sigmoid squashes output to [0, 1], perfect for binary classification
    Dense(units=1, activation='sigmoid')
])

# Compile with binary cross-entropy (standard for 2-class classification)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# =============================================================================
# 5. Train the Model
# =============================================================================
# With a small dataset:
# - Use small batch_size (2) so we update weights more frequently
# - Use validation_split to monitor overfitting
# - More epochs are okay since the dataset is tiny

epochs = 20
batch_size = 2

print("\nTraining the model...")
model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)


# =============================================================================
# 6. Make Predictions
# =============================================================================
# model.predict() returns probabilities (0 to 1)
# We convert to binary predictions using 0.5 threshold
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)


# =============================================================================
# 7. Evaluate Model Performance
# =============================================================================
print("\n" + "=" * 50)
print("Classification Report on Test Set")
print("=" * 50)
print(classification_report(y_test, y_pred, zero_division=0))


# =============================================================================
# 8. Test on a New Review
# =============================================================================
# Demonstrate how to use the trained model on completely new text
new_review = ["This was an average movie, neither good nor bad."]

# Same preprocessing as training data
new_sequence = tokenizer.texts_to_sequences(new_review)
padded_new_sequence = pad_sequences(new_sequence, maxlen=maxlen, padding='post', truncating='post')

# Get prediction
prediction_prob = model.predict(padded_new_sequence)[0][0]
prediction_label = "Positive" if prediction_prob > 0.5 else "Negative"

print("=" * 50)
print("Predicting on New Review")
print("=" * 50)
print(f"Review: '{new_review[0]}'")
print(f"Predicted probability: {prediction_prob:.4f}")
print(f"Predicted sentiment: {prediction_label}")
print()
print("Note: With such a small training set, predictions may not be reliable!")