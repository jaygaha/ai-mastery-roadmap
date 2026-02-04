"""
Advanced DL Module - Example 2: Multimodal Learning with Embeddings & LSTM

This script takes a more advanced approach:
1.  **Multimodal Learning**: We treat "numerical data" and "text data" as two separate inputs (like eyes and ears).
2.  **Word Embeddings**: Instead of counting words (TF-IDF), we learn a "meaning vector" for each word.
3.  **LSTM (Long Short-Term Memory)**: We use a Recurrent Neural Network to read the text *sequence* from start to finish, capturing context (e.g., "not happy" is different from "happy").

Architecture:
[Measurements] -> [Dense Layers] -----\
                                       (+) -> [Combined] -> [Prediction]
[Feedback Text] -> [Embedding] -> [LSTM] -/
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, LSTM, GlobalAveragePooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# --- 1. Data Simulation ---
np.random.seed(42)
num_samples = 1000
churn_df = pd.DataFrame({
    'customer_id': range(num_samples),
    'MonthlyCharges': np.random.rand(num_samples) * 100,
    'TotalCharges': np.random.rand(num_samples) * 2000,
    'Tenure': np.random.randint(1, 72, num_samples),
    'Gender': np.random.choice(['Male', 'Female'], num_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples),
    'IsSeniorCitizen': np.random.randint(0, 2, num_samples),
    'Churn': np.random.randint(0, 2, num_samples)
})

# Simulate customer feedback
feedback_positive = [
    "Great service, very happy!", "Product works perfectly, love it.", "Excellent support, solved my issue quickly.",
    "Very satisfied with the features.", "Highly recommend this company.", "Fantastic experience overall.",
    "No complaints, everything is smooth.", "Happy customer, will stay with them."
]
feedback_negative = [
    "Terrible customer service, very slow.", "Unhappy with the recent update, buggy.", "Constant outages, frustrated.",
    "Billing issues, very annoying.", "Considering canceling my subscription.", "Poor value for money.",
    "Frustrated with the lack of features.", "Wouldn't recommend to anyone."
]
feedback_neutral = [
    "It's okay, nothing special.", "Service is adequate.", "Standard experience.", "No strong feelings either way.",
    "Works as expected.", "Average performance.", "Could be better, could be worse.", "Just fine."
]

customer_feedback = []
for i in range(num_samples):
    if churn_df.loc[i, 'Churn'] == 1:
        feedback_type = np.random.choice(['negative', 'neutral', 'positive'], p=[0.6, 0.3, 0.1])
    else:
        feedback_type = np.random.choice(['positive', 'neutral', 'negative'], p=[0.6, 0.3, 0.1])

    if feedback_type == 'positive':
        customer_feedback.append(np.random.choice(feedback_positive))
    elif feedback_type == 'negative':
        customer_feedback.append(np.random.choice(feedback_negative))
    else:
        customer_feedback.append(np.random.choice(feedback_neutral))

churn_df['customer_feedback'] = customer_feedback

# --- 2. Preprocessing Standard Features ---
numerical_cols = ['MonthlyCharges', 'TotalCharges', 'Tenure']
categorical_cols = ['Gender', 'Contract', 'IsSeniorCitizen']

churn_df = pd.get_dummies(churn_df, columns=categorical_cols, drop_first=True)

for col in numerical_cols:
    min_val = churn_df[col].min()
    max_val = churn_df[col].max()
    churn_df[col] = (churn_df[col] - min_val) / (max_val - min_val)

X_numerical_categorical = churn_df.drop(columns=['customer_id', 'customer_feedback', 'Churn']).astype('float32').values
y = churn_df['Churn'].values

# --- 3. Preprocessing Text Features for Embedding ---
text_data = churn_df['customer_feedback'].values

# A. Tokenization: Convert words to numbers (Indices)
# 'num_words=1000': Vocabulary size. Only top 1000 words matter; others become 'unknown'.
tokenizer = Tokenizer(num_words=1000, oov_token="<unk>") 
tokenizer.fit_on_texts(text_data)
word_index = tokenizer.word_index

# B. Sequencing: Convert text sentences to lists of numbers
sequences = tokenizer.texts_to_sequences(text_data)

# C. Padding: Ensure all sentences are the same length for the Neural Network.
# 'post': Add zeros at the end if the sentence is too short.
max_sequence_length = 20 
X_text_padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Split data (We need to split BOTH the numerical X and the text X)
X_num_cat_train, X_num_cat_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_numerical_categorical, X_text_padded, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Build Multimodal Model (Functional API) ---
# The Functional API allows for complex topologies (like two inputs merging into one).

# Branch 1: Standard Numerical Inputs
num_cat_input = Input(shape=(X_num_cat_train.shape[1],), name='num_cat_input')
num_cat_branch = Dense(64, activation='relu')(num_cat_input)
num_cat_branch = Dense(32, activation='relu')(num_cat_branch)

# Branch 2: Text Inputs
text_input = Input(shape=(max_sequence_length,), name='text_input')
vocab_size = len(word_index) + 1 # +1 for the padding token (0)
embedding_dim = 50 

# Embedding Layer: Learns a 50-dimensional vector for every word in our vocabulary
text_branch = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)

# LSTM Layer: Reads the sequence of 50-D word vectors. 
# It keeps a "memory" of what it has read so far.
text_branch = LSTM(64)(text_branch) 

# Branch 3: Merging
# We concatenate the output vector of the Numerical MLP and the Text LSTM
merged = Concatenate()([num_cat_branch, text_branch])

# Final Classification Layers
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.3)(merged)
merged = Dense(32, activation='relu')(merged)
merged = Dropout(0.3)(merged)
output = Dense(1, activation='sigmoid')(merged)

model_multimodal = Model(inputs=[num_cat_input, text_input], outputs=output)

model_multimodal.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 5. Train Model ---
# Note: We pass a DICTIONARY of inputs matching the names of our Input layers
model_multimodal.fit(
    {'num_cat_input': X_num_cat_train, 'text_input': X_text_train},
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)

loss, accuracy = model_multimodal.evaluate(
    {'num_cat_input': X_num_cat_test, 'text_input': X_text_test},
    y_test,
    verbose=0
)
print(f"Multimodal Model (Embeddings+LSTM) - Test Accuracy: {accuracy:.4f}")