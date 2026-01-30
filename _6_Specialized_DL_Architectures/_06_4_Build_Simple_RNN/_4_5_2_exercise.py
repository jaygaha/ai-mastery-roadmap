"""
Exercise 2 Solution: Text Classification with More Data and Enhanced Features

This exercise explores how dataset size and model capacity affect RNN performance.

Objective:
    1. Add more positive and negative sentences to the dataset
    2. Increase embedding_dim to 32 (from 16)
    3. Increase SimpleRNN units to 64 (from 32)
    4. Observe how these changes affect training accuracy

Key learning points:
    - More training data generally leads to better generalization
    - Larger embedding dimensions can capture more semantic nuance
    - More RNN units increase model capacity (can learn more complex patterns)
    - Trade-off: Larger models need more data to avoid overfitting

Expected observations:
    - Training accuracy should improve with more data and capacity
    - Watch for overfitting: high training accuracy but low validation accuracy
    - The model may still struggle with truly ambiguous sentiments
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
# 1. Expanded Text Data
# =============================================================================
# ========================== EXERCISE MODIFICATION ============================
# Added more positive and negative reviews to increase dataset size
# Original dataset had 12 reviews; now we have 14 reviews
reviews = [
    # Original positive reviews (4)
    "This movie was fantastic and I loved it!",
    "I really enjoyed the plot and the acting was superb.",
    "A wonderful film, highly recommend.",
    "Absolutely brilliant, a must-watch.",
    
    # Added positive reviews (3)
    "The director's vision was inspiring, truly captivating.",
    "An uplifting and heartwarming story, beautifully told.",
    "I'll watch this again, pure cinematic joy!",

    # Original negative reviews (4)
    "Terrible movie, very boring and a waste of time.",
    "I hated every minute of it, utterly dreadful.",
    "Not good, quite disappointing.",
    "Could have been better, somewhat dull.",
    
    # Added negative reviews (3)
    "The plot was confusing and the acting was wooden.",
    "A truly unbearable experience, do not recommend.",
    "Worst film of the year, a complete disaster."
]

# Updated labels to match the expanded review list
# 7 positive (1) + 7 negative (0) = 14 total
sentiments = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
# =============================================================================

print(f"Dataset size: {len(reviews)} reviews")
print(f"Positive: {sum(sentiments)}, Negative: {len(sentiments) - sum(sentiments)}")


# =============================================================================
# 2. Tokenize and Vectorize Text Data
# =============================================================================
tokenizer = Tokenizer(num_words=1000, oov_token="<unk>")
tokenizer.fit_on_texts(reviews)

sequences = tokenizer.texts_to_sequences(reviews)

maxlen = max(len(s) for s in sequences)
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

labels = np.array(sentiments)

print(f"Vocabulary size: {len(tokenizer.word_index)} unique words")
print(f"Max sequence length: {maxlen}")


# =============================================================================
# 3. Split Data into Training and Testing Sets
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


# =============================================================================
# 4. Build the Enhanced Simple RNN Model
# =============================================================================
vocab_size = len(tokenizer.word_index) + 1

# ========================== EXERCISE MODIFICATION ============================
# Increased embedding_dim from 16 to 32
# Why? Larger embeddings can capture more nuanced word relationships
embedding_dim = 32

model_text = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    
    # Increased SimpleRNN units from 32 to 64
    # Why? More units = larger hidden state = can remember more information
    SimpleRNN(units=64),
    
    Dense(units=1, activation='sigmoid')
])
# =============================================================================

model_text.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_text.summary()


# =============================================================================
# 5. Train the Model
# =============================================================================
epochs = 20
batch_size = 2

print("\nTraining the model...")
history = model_text.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)


# ========================== EXERCISE ADDITION ================================
# Analyze how the enhanced configuration affects training
print("\n" + "=" * 50)
print(f"Training Results (embedding_dim={embedding_dim}, RNN_units=64)")
print("=" * 50)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print()
print("Compare with original settings (embedding_dim=16, units=32):")
print("  - Higher capacity may lead to faster convergence")
print("  - Watch the gap between training/validation accuracy for overfitting")
# =============================================================================


# =============================================================================
# 6. Make Predictions
# =============================================================================
y_pred_probs = model_text.predict(X_test)
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
new_review = ["This was an average movie, neither good nor bad, but I still enjoyed it."]
new_sequence = tokenizer.texts_to_sequences(new_review)
padded_new_sequence = pad_sequences(new_sequence, maxlen=maxlen, padding='post', truncating='post')

prediction_prob = model_text.predict(padded_new_sequence)[0][0]
prediction_label = "Positive" if prediction_prob > 0.5 else "Negative"

print("=" * 50)
print("Predicting on New Review")
print("=" * 50)
print(f"Review: '{new_review[0]}'")
print(f"Predicted probability: {prediction_prob:.4f}")
print(f"Predicted sentiment: {prediction_label}")
print()
print("Note: Even with more data, small datasets can still produce unreliable predictions!")