
# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# print(tf.__version__)
# Expected output: A version number like '2.18.0' or similar

"""
Data Preparation
"""
# Load the real dataset from the previous module
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
    print("Error: 'Telco-Customer-Churn.csv' not found. Please ensure the data file exists.")
    print("Using synthetic data for demonstration ONLY.")
    # Fallback to synthetic data
    np.random.seed(42)
    data_size = 1000
    df = pd.DataFrame({
        'feature_1': np.random.rand(data_size),
        'feature_2': np.random.rand(data_size) * 10,
        'feature_3': np.random.randint(0, 5, data_size),
        'churn': np.random.randint(0, 2, data_size)
    })

# Define features (X) and target (y)
target_col = 'Churn' if 'Churn' in df.columns else 'churn'
if target_col not in df.columns:
     target_col = df.columns[-1]

X = df.drop(target_col, axis=1)
y = df[target_col]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame (optional, but good for inspection)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("Shape of X_train_scaled:", X_train_scaled.shape)
print("Shape of y_train:", y_train.shape)
print("First 5 rows of scaled training features:")
print(X_train_scaled_df.head())


"""
Building the Neural Network
"""

# Initialize the Sequential model
model = keras.Sequential()

# Add the input layer and the first hidden layer
# 'input_shape' specifies the number of features in the input data.
model.add(layers.Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],))) 

# Add another hidden layer
model.add(layers.Dense(units=32, activation='relu')) 

# Add the output layer
# For binary classification, a single neuron with a 'sigmoid' activation is used.
model.add(layers.Dense(units=1, activation='sigmoid'))

# Display the model's architecture
model.summary()

"""
Compiling the Model
"""

# Compile the model
# optimizer: 'adam' is efficient for most cases.
# loss: 'binary_crossentropy' is standard for binary classification.
# metrics: 'accuracy' to monitor performance.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model compiled successfully!")

"""
Train the model
"""
history = model.fit(X_train_scaled, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1, # Use 10% of training data for validation
                    verbose=1) 

"""
Evaluating the model
"""
# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions on the test set
y_pred_proba = model.predict(X_test_scaled) # Outputs probabilities
y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary class labels

# Calculate additional classification metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")