"""
Exercise 2-4 Solution: Architecture, Dimensions, and Hyperparameters

Goal: Experiment with the Multimodal Model settings.
1.  **Architecture**: Replace LSTM with GlobalAveragePooling1D (faster, but no sequence awareness).
2.  **Embedding Dimensions**: Reduce to 20 (faster training, less semantic nuance).
3.  **Hyperparameters**: Increase batch size to 64 and epochs to 20 (investigating convergence and overfitting).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, LSTM, GlobalAveragePooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# --- Data Simulation ---
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
feedback_positive = ["Great service", "Love it", "Excellent", "Satisfied", "Recommend", "Fantastic", "Smooth", "Happy"]
feedback_negative = ["Terrible", "Unhappy", "Outages", "Billing issues", "Canceling", "Poor value", "Frustrated", "Bad"]
feedback_neutral = ["Okay", "Adequate", "Standard", "Fine", "Average", "So-so", "Normal", "Expected"]

customer_feedback = []
for i in range(num_samples):
    if churn_df.loc[i, 'Churn'] == 1:
        feedback_type = np.random.choice(['negative', 'neutral', 'positive'], p=[0.6, 0.3, 0.1])
    else:
        feedback_type = np.random.choice(['positive', 'neutral', 'negative'], p=[0.6, 0.3, 0.1])
    
    if feedback_type == 'positive': customer_feedback.append(np.random.choice(feedback_positive))
    elif feedback_type == 'negative': customer_feedback.append(np.random.choice(feedback_negative))
    else: customer_feedback.append(np.random.choice(feedback_neutral))

churn_df['customer_feedback'] = customer_feedback

# Preprocessing Numerical/Categorical
numerical_cols = ['MonthlyCharges', 'TotalCharges', 'Tenure']
categorical_cols = ['Gender', 'Contract', 'IsSeniorCitizen']
churn_df = pd.get_dummies(churn_df, columns=categorical_cols, drop_first=True)

for col in numerical_cols:
    min_val = churn_df[col].min()
    max_val = churn_df[col].max()
    churn_df[col] = (churn_df[col] - min_val) / (max_val - min_val)

X_numerical_categorical = churn_df.drop(columns=['customer_id', 'customer_feedback', 'Churn']).astype('float32').values
y = churn_df['Churn'].values

# Preprocessing Text
text_data = churn_df['customer_feedback'].values
tokenizer = Tokenizer(num_words=1000, oov_token="<unk>")
tokenizer.fit_on_texts(text_data)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(text_data)
max_sequence_length = 20
X_text_padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

X_num_cat_train, X_num_cat_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_numerical_categorical, X_text_padded, y, test_size=0.2, random_state=42, stratify=y
)

# --- Build Multimodal Model ---
num_cat_input = Input(shape=(X_num_cat_train.shape[1],), name='num_cat_input')
num_cat_branch = Dense(64, activation='relu')(num_cat_input)
num_cat_branch = Dense(32, activation='relu')(num_cat_branch)

text_input = Input(shape=(max_sequence_length,), name='text_input')
vocab_size = len(word_index) + 1

# --- Exercise 3: Embedding Dimension ---
# Changed embedding_dim from 50 to 20.
# Trade-off: Lower dimension = faster training, less memory, less capacity for semantic nuances.
embedding_dim = 20 
text_branch = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)

# --- Exercise 2: Text Branch Architecture ---
# Replaced LSTM with GlobalAveragePooling1D.
# comparison:
# - LSTM: Good for sequential dependencies, context awareness. Slower.
# - GlobalAveragePooling1D: Averages embeddings across the sequence. Fast, invariant to word order (bag-of-words vibe). Good if specific keywords drive the sentiment regardless of order.
text_branch = GlobalAveragePooling1D()(text_branch) 
# text_branch = LSTM(64)(text_branch) # Previous implementation

merged = Concatenate()([num_cat_branch, text_branch])
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.3)(merged)
merged = Dense(32, activation='relu')(merged)
merged = Dropout(0.3)(merged)
output = Dense(1, activation='sigmoid')(merged)

model_multimodal = Model(inputs=[num_cat_input, text_input], outputs=output)
model_multimodal.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Exercise 4: Hyperparameter Tuning ---
# Experimenting with batch_size=64 and epochs=20
# Tuning comments:
# - Smaller batch size often generalizes better but is noisier.
# - More epochs allow convergence but risk overfitting.
# - Regularization: If overfitting occurs, increase Dropout rates or add L2 regularization (kernel_regularizer).
history = model_multimodal.fit(
    {'num_cat_input': X_num_cat_train, 'text_input': X_text_train},
    y_train,
    epochs=20,     # Changed from 10
    batch_size=64, # Changed from 32
    validation_split=0.1,
    verbose=0
)

loss, accuracy = model_multimodal.evaluate(
    {'num_cat_input': X_num_cat_test, 'text_input': X_text_test},
    y_test,
    verbose=0
)
print(f"Exercise 2-4 Result (GAP1D, dim=20, epochs=20) - Test Accuracy: {accuracy:.4f}")
