"""
Advanced DL Module - Example 1: Associating Text Features via TF-IDF

This script demonstrates a simple multimodal approach:
1.  Simulate a customer churn dataset with standard features (numerical/categorical) AND textual feedback.
2.  Preprocess the text using TF-IDF (Term Frequency-Inverse Document Frequency), which converts text into valid numerical vectors based on word usage.
3.  Train a Deep Learning model that uses both the standard customer profile and the sentiment from their feedback to predict churn.

Key Concept: TF-IDF is a "bag-of-words" approach. It doesn't care about the order of words, just their presence and rarity.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
import numpy as np

# --- 1. Data Simulation ---
# Creating a dummy dataset to mimic real-world telecom churn data
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
    'Churn': np.random.randint(0, 2, num_samples) # Target variable
})

# Simulate customer feedback (textual data)
# We artificially link negative feedback to churners to give the model a pattern to learn.
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

# Randomly assign feedback to customers
customer_feedback = []
for i in range(num_samples):
    if churn_df.loc[i, 'Churn'] == 1:
        # Churners are more likely to have negative feedback
        feedback_type = np.random.choice(['negative', 'neutral', 'positive'], p=[0.6, 0.3, 0.1])
    else:
        # Non-churners are more likely to have positive feedback
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

# One-hot encode categorical features (convert 'Male'/'Female' to 0/1 columns)
churn_df = pd.get_dummies(churn_df, columns=categorical_cols, drop_first=True)

# Scale numerical features (0-1 range) to help the neural network converge faster
for col in numerical_cols:
    min_val = churn_df[col].min()
    max_val = churn_df[col].max()
    churn_df[col] = (churn_df[col] - min_val) / (max_val - min_val)

# Separate features (X) and target (y)
# IMPORTANT: Casting to float32 to ensure compatibility with TensorFlow
X_numerical_categorical = churn_df.drop(columns=['customer_id', 'customer_feedback', 'Churn']).astype('float32').values
y = churn_df['Churn'].values

# --- 3. Preprocessing Text Features (The New Part!) ---
# TF-IDF Vectorization:
# - Counts identifying words.
# - 'max_features=100': We only keep the top 100 most important words to keep the model simple.
tfidf_vectorizer = TfidfVectorizer(max_features=100) 
X_text_tfidf = tfidf_vectorizer.fit_transform(churn_df['customer_feedback']).toarray()

# --- 4. Combining Features ---
# We simply glue the TF-IDF vectors onto the end of our standard feature vectors.
X_combined = np.concatenate((X_numerical_categorical, X_text_tfidf), axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Build and Train Model ---
input_dim = X_train.shape[1]

model = Sequential([
    # We use an explicit Input layer to define the shape of our combined data
    Input(shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dropout(0.3), # Dropout helps prevent overfitting to specific words
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification (Churn yes/no)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Model with TF-IDF text features - Test Accuracy: {accuracy:.4f}")