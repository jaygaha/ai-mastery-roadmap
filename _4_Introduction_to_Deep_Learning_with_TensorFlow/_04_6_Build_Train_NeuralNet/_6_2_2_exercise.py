"""
Exercises

2. Experiment with Training Parameters: Using the original model architecture (64, 32 neurons), experiment with:
    - Epochs: Change epochs from 50 to 100 or 20. How does the model's performance on the validation set change? Does it improve or worsen?
    - Batch Size: Change batch_size from 32 to 16 or 64. Observe the impact on training speed and final accuracy.
"""

"""
SOLUTION

This exercise explores how training hyperparameters like the number of epochs and batch size affect the model's learning process and final performance.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Data Loading and Preprocessing (Same as Exercise 1) ---

try:
    df = pd.read_csv('../../_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling/Telco-Customer-Churn.csv')
    
    # Preprocessing
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    df = pd.get_dummies(df, drop_first=True)

except FileNotFoundError:
    print("Error: Dataset not found. Generating dummy data for demonstration.")
    data_size = 1000
    df = pd.DataFrame({
        'feature_1': np.random.rand(data_size),
        'feature_2': np.random.rand(data_size) * 10,
        'feature_3': np.random.randint(0, 5, data_size),
        'churn': np.random.randint(0, 2, data_size)
    })
    # Add dummy columns to match expected shape if needed, or just proceed

# Separate features and target
target_col = 'Churn' if 'Churn' in df.columns else 'churn'
if target_col not in df.columns:
     target_col = df.columns[-1]

X = df.drop(target_col, axis=1)
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Data prepared. Input shape: {X_train_scaled.shape}")


# --- Exercise 2: Different Training Parameters ---

print("\n--- Exercise 2: Different Training Parameters ---")

# Original model architecture function to easily recreate the model
def create_original_model():
    model = keras.Sequential([
        layers.Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dense(units=32, activation='relu'),
        layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Configuration A: More Epochs (100)
print("\n[Configuration A] More Epochs (100)")
# We use the original architecture
model_epochs_100 = create_original_model()

# Note: Using verbose=0 to reduce output, but you can set to 1 to see progress
history_epochs_100 = model_epochs_100.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
loss_e100, acc_e100 = model_epochs_100.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss (100 epochs): {loss_e100:.4f}, Test Accuracy (100 epochs): {acc_e100:.4f}")


# Configuration B: Smaller Batch Size (16)
print("\n[Configuration B] Smaller Batch Size (16)")
# Re-initialize model for batch size experiment to avoid retraining on previous epochs
model_batch_16 = create_original_model()

history_batch_16 = model_batch_16.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0)
loss_b16, acc_b16 = model_batch_16.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss (batch 16): {loss_b16:.4f}, Test Accuracy (batch 16): {acc_b16:.4f}")

print("\n--- Experimentation Complete ---")