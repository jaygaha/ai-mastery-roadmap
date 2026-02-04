"""
Exercise 1 Solution: Tuning TF-IDF parameters

Goal: Investigate how changing the vocabulary size (max_features) affects model performance.
- We increased max_features from 100 to 200.
- Hypotheses: More words = more information, but also more noise and higher risk of overfitting.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
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

# --- EXERCISE MODIFICATION HERE ---
# Changed max_features from 100 to 200
# Trade-off Question: 
# - Small max_features: Less computational cost, faster training, but might miss important rare words.
# - Large max_features: captures more vocabulary, potentially higher accuracy, but increases model size and risk of overfitting (curse of dimensionality).
tfidf_vectorizer = TfidfVectorizer(max_features=200) 
X_text_tfidf = tfidf_vectorizer.fit_transform(churn_df['customer_feedback']).toarray()

X_combined = np.concatenate((X_numerical_categorical, X_text_tfidf), axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

input_dim = X_train.shape[1]

model = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Exercise 1 Result (max_features=200) - Test Accuracy: {accuracy:.4f}")
