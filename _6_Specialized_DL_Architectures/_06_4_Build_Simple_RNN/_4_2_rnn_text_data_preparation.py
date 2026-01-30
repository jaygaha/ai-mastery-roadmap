"""
Text Data Preparation for RNNs

This script demonstrates how to prepare text data for training a Recurrent Neural
Network. Unlike numbers, text must be converted into a numerical format that
neural networks can process.

The three key steps are:
    1. Tokenization: Split text into individual words (tokens)
    2. Numericalization: Convert each word to a unique integer ID
    3. Padding: Ensure all sequences have the same length

Example transformation:
    Text: "I love cats"
    Step 1 (Tokenize): ["I", "love", "cats"]
    Step 2 (Numericalize): [5, 12, 8]  (IDs depend on vocabulary)
    Step 3 (Pad to length 5): [5, 12, 8, 0, 0]

Why padding? RNNs process batches of sequences, and batch processing requires
all sequences in a batch to have the same length. Padding adds zeros (a special
"nothing" token) to shorter sequences.
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# =============================================================================
# Example Text Data for Sentiment Classification
# =============================================================================
# A tiny dataset of movie reviews with positive (1) and negative (0) labels.
# In real applications, you'd have thousands of examples.
texts = [
    "I love this movie, it's fantastic!",
    "This movie was terrible and boring.",
    "A wonderful film with great actors.",
    "I hated every minute of it."
]
labels = np.array([1, 0, 1, 0])  # 1 = positive sentiment, 0 = negative sentiment


# =============================================================================
# Step 1 & 2: Tokenization and Numericalization
# =============================================================================
# The Tokenizer handles both steps:
#   - Learns a vocabulary from the training texts
#   - Converts words to integer IDs

# Create a tokenizer
# - num_words=None: Keep all words (no vocabulary limit)
# - oov_token="<unk>": Replace unknown words with this token
#   (Words seen during inference but not in training vocabulary)
tokenizer = Tokenizer(num_words=None, oov_token="<unk>")

# Build the vocabulary from our texts
# This creates a word-to-ID mapping like {"love": 5, "movie": 3, ...}
tokenizer.fit_on_texts(texts)

# Convert each text to a sequence of integers
# "I love this movie" → [word_id_for_I, word_id_for_love, ...]
sequences = tokenizer.texts_to_sequences(texts)

print("=" * 50)
print("Step 1 & 2: Tokenization and Numericalization")
print("=" * 50)
print(f"Original texts: {texts[0][:30]}...")
print(f"Converted to sequence of IDs: {sequences[0]}")
print()


# =============================================================================
# Step 3: Padding
# =============================================================================
# Determine the maximum sequence length in our data
max_sequence_length = max(len(seq) for seq in sequences)

# Pad sequences to uniform length
# - maxlen: Target length for all sequences
# - padding='post': Add zeros at the END of shorter sequences
#   (Alternative: 'pre' adds zeros at the BEGINNING)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

print("Step 3: Padding")
print("=" * 50)
print("Padded sequences (each row is one text):")
print(padded_sequences)
print()
print(f"Vocabulary size: {len(tokenizer.word_index)} unique words")
print(f"Max sequence length: {max_sequence_length}")
print()


# =============================================================================
# Understanding the Output Shape
# =============================================================================
# At this point, padded_sequences has shape: (num_samples, max_sequence_length)
# This is (4, 6) in our example: 4 texts, each with 6 integer IDs.
#
# However, RNNs expect input shape: (samples, timesteps, features)
# The Embedding layer (used in actual RNN models) handles this:
#   Input:  (4, 6) - 4 samples, 6 tokens each
#   Output: (4, 6, embedding_dim) - each token becomes a dense vector
#
# The embedding dimension (e.g., 16 or 32) determines how "rich" each word's
# representation is. Similar words will have similar embedding vectors.

print("Ready for RNN!")
print("=" * 50)
print(f"Shape for Embedding layer input: {padded_sequences.shape}")
print(f"After Embedding (with dim=16): would be ({padded_sequences.shape[0]}, {padded_sequences.shape[1]}, 16)")
print("This matches RNN's expected (batch_size, timesteps, features) format!")