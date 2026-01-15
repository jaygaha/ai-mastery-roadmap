import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Assume X_train, X_test, y_train, y_test are already prepared
# from the Customer Churn Prediction Case Study (Module 2, 3)

# For demonstration, let's create dummy data similar to the churn case study
# 10 features, 1000 samples for a binary classification problem
np.random.seed(42)
X = np.random.rand(1000, 10) * 100
y = (np.random.rand(1000) > 0.5).astype(int) # Binary target

# Simulate feature scaling as done in Module 2
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the MLP model
model = keras.Sequential([
    # Input layer implicitly defined by input_shape in the first hidden layer
    keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)), # First hidden layer
    keras.layers.Dense(units=32, activation='relu'), # Second hidden layer
    keras.layers.Dense(units=1, activation='sigmoid') # Output layer for binary classification
])

# Display model summary
model.summary()

# Compile the model
# Compile the model
# For binary classification: 'binary_crossentropy' loss
# For regression: 'mean_squared_error' or 'mean_absolute_error'
# For multi-class classification: 'categorical_crossentropy' or 'sparse_categorical_crossentropy'

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) # For classification tasks

# If it were a regression problem:
# model_regression = keras.Sequential([
#     keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),
#     keras.layers.Dense(units=32, activation='relu'),
#     keras.layers.Dense(units=1) # Linear activation for regression output
# ])
# model_regression.compile(optimizer='adam',
#                          loss='mean_squared_error',
#                          metrics=['mae']) # Mean Absolute Error for regression

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# The 'history' object contains training loss and metrics for each epoch
print(history.history.keys())

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions on new data (e.g., X_test)
y_pred_proba = model.predict(X_test)

# For binary classification, convert probabilities to class labels
y_pred_classes = (y_pred_proba > 0.5).astype(int)

# Display some predictions
print("Sample Predictions (Probabilities):")
print(y_pred_proba[:5].flatten())
print("Sample Predicted Classes:")
print(y_pred_classes[:5].flatten())

# If it were a regression problem:
# y_pred_regression = model_regression.predict(X_test)
# print("Sample Regression Predictions:")
# print(y_pred_regression[:5].flatten())