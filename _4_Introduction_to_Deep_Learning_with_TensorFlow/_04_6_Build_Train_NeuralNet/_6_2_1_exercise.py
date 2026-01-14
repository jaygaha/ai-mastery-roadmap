
"""
Exercises

1. Experiment with Network Architecture: Modify the `build_and_train_model` function.
    - Change the number of neurons in the hidden layers (e.g., from 64/32 to 128/64 or 32/16).
    - Add a third hidden layer or remove one hidden layer.
    - Observe how these changes affect the model.summary() output (especially the number of parameters) and the final test accuracy.
"""

"""
SOLUTION

This exercise focuses on understanding how different architectural choices impact a simple feedforward neural network's performance on the customer churn prediction dataset. 
We'll build and train several models, varying the number of layers and neurons per layer, and analyze the results.

Setup

First, ensure you have the necessary libraries and the prepared customer churn data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the preprocessed data (assuming it's available from previous modules)
# Replace 'customer_churn_preprocessed.csv' with your actual file path if different
try:
    df = pd.read_csv('../../_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling/Telco-Customer-Churn.csv')
    
    # Preprocessing
    # Drop customerID as it's not a feature
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # Convert TotalCharges to numeric (coercing errors to NaN), then fill NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Encode target variable 'Churn' (Yes/No -> 1/0)
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

except FileNotFoundError:
    print("Error: 'customer_churn_preprocessed.csv' not found. Please ensure your preprocessed data is in the correct directory.")
    print("For demonstration, generating dummy data...")
    # Generate dummy data for demonstration if file not found
    data_size = 1000
    df = pd.DataFrame({
        'feature_1': np.random.rand(data_size),
        'feature_2': np.random.rand(data_size) * 10,
        'feature_3': np.random.randint(0, 5, data_size),
        'feature_4': np.random.normal(50, 10, data_size),
        'gender_Female': np.random.randint(0, 2, data_size),
        'gender_Male': np.random.randint(0, 2, data_size),
        'has_credit_card': np.random.randint(0, 2, data_size),
        'is_active_member': np.random.randint(0, 2, data_size),
        'churn': np.random.randint(0, 2, data_size)
    })
    df['gender_Male'] = 1 - df['gender_Female'] # Ensure one-hot logic
    df['feature_5'] = np.random.rand(data_size) # Add more features for a realistic network input

# Separate features (X) and target (y)
# Check for 'Churn' or 'churn' (lowercase) depending on source
target_col = 'Churn' if 'Churn' in df.columns else 'churn'
if target_col not in df.columns:
     # Fallback if neither found, though preprocess usually ensures Churn
    print("Warning: Churn column not found, using last column as target")
    target_col = df.columns[-1]

X = df.drop(target_col, axis=1)
y = df[target_col]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get input shape for the network
input_shape = X_train_scaled.shape[1]
print(f"Input features: {input_shape}")


def build_and_train_model(X_train, y_train, X_test, y_test, num_hidden_layers=2, units_per_layer=[64, 32], epochs=20, batch_size=32):
    """
    Builds, compiles, trains, and evaluates a simple feedforward neural network.
    
    Args:
        X_train, y_train: Training data.
        X_test, y_test: Testing data.
        num_hidden_layers: Number of hidden layers to add.
        units_per_layer: List containing the number of neurons for each hidden layer.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        
    Returns:
        model: The trained Keras model.
        history: The training history object.
    """
    
    model = keras.Sequential()
    
    # Input layer calling the first hidden layer
    # We use input_shape argument in the first layer to define input dimensions
    if num_hidden_layers > 0:
        model.add(layers.Dense(units=units_per_layer[0], activation='relu', input_shape=(X_train.shape[1],)))
    else:
        # Fallback if no hidden layers (basically logistic regression)
         model.add(layers.Dense(units=1, activation='sigmoid', input_shape=(X_train.shape[1],)))
         
    # Additional hidden layers
    for i in range(1, num_hidden_layers):
        # Use provided units if available, otherwise default to half of previous or 16
        if i < len(units_per_layer):
            units = units_per_layer[i]
        else:
            units = units_per_layer[-1] // 2 if units_per_layer else 32
        
        model.add(layers.Dense(units=max(units, 4), activation='relu')) # Ensure at least 4 neurons

    # Output layer
    # Sigmoid activation for binary classification (0 or 1)
    if num_hidden_layers > 0:
        model.add(layers.Dense(units=1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    print(f"\nModel Summary (Layers: {num_hidden_layers}, Units: {units_per_layer}):")
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_split=0.1, 
                        verbose=0) # verbose=0 to reduce clutter, we print result at end

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    return model, history

# --- Exercise Execution ---

print("\n--- Exercise 1: Experiment with Network Architectures ---")

# Configuration 1: Two hidden layers (128, 64 neurons)
# This increase capacity should allow learning more complex patterns, 
# but risks overfitting if data is small.
print("\n[Configuration 1] Two hidden layers: 128, 64 neurons")
model1, history1 = build_and_train_model(X_train_scaled, y_train, X_test_scaled, y_test,
                                         num_hidden_layers=2, units_per_layer=[128, 64], epochs=15)

# Configuration 2: One hidden layer (32 neurons)
# A simpler model. Validates if a complex structure is actually necessary.
print("\n[Configuration 2] One hidden layer: 32 neurons")
model2, history2 = build_and_train_model(X_train_scaled, y_train, X_test_scaled, y_test,
                                         num_hidden_layers=1, units_per_layer=[32], epochs=15)

# Configuration 3: Three hidden layers (64, 32, 16 neurons)
# Deeper network. Sometimes depth helps with hierarchical feature learning.
print("\n[Configuration 3] Three hidden layers: 64, 32, 16 neurons")
model3, history3 = build_and_train_model(X_train_scaled, y_train, X_test_scaled, y_test,
                                         num_hidden_layers=3, units_per_layer=[64, 32, 16], epochs=15)

print("\n--- Experimentation Complete ---")
print("Compare the Test Accuracy and number of parameters (Total params) above.")