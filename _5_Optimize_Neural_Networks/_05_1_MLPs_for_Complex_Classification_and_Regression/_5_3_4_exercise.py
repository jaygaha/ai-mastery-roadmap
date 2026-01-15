"""
4. Regression Model with Different Metrics: Using the regression example, modify the model_reg.compile() call to include 'mse' and 'rmse' 
    (Root Mean Squared Error, which you'll need to calculate manually or define as a custom metric for Keras, though for this exercise, 
    simply include 'mse' and note that RMSE is the square root of MSE).

    - Explain why mae might sometimes be preferred over mse for certain regression tasks, especially when outliers are present. 
        (Hint: Recall Module 3 on model evaluation metrics).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_regression # For generating synthetic regression data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# --- 1. Generate a Synthetic Regression Dataset ---
print("--- Generating Synthetic Regression Data ---")
n_samples = 1500
n_features = 10

X, y = make_regression(n_samples=n_samples,
                       n_features=n_features,
                       n_informative=7,
                       noise=20, # Add some noise to make it realistic
                       random_state=42)

print(f"Generated X shape: {X.shape}")
print(f"Generated y shape: {y.shape}")
print(f"Sample y values: {y[:10]}")

# --- 2. Preprocess Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = tf.constant(X_train_scaled, dtype=tf.float32)
X_test_tensor = tf.constant(X_test_scaled, dtype=tf.float32)
y_train_tensor = tf.constant(y_train, dtype=tf.float32)
y_test_tensor = tf.constant(y_test, dtype=tf.float32)

input_dim = X_train_scaled.shape[1]

# --- 3. Build and Train an MLP for Regression ---
print("\n--- Building and Training Regression MLP ---")
model_regression = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu', name='hidden_layer_1'),
    layers.Dense(32, activation='relu', name='hidden_layer_2'),
    layers.Dense(1, activation='linear', name='output_layer_linear') # 1 neuron for single output, linear activation
])

model_regression.compile(optimizer='adam',
                         loss='mean_squared_error', # Common loss for regression
                         metrics=['mean_absolute_error']) # Also monitor MAE

print(model_regression.summary())

history_regression = model_regression.fit(X_train_tensor, y_train_tensor,
                                          epochs=150, # Regression often benefits from more epochs
                                          batch_size=32,
                                          validation_split=0.1,
                                          verbose=1)

# --- 4. Evaluate the Model with Regression Metrics ---
print("\n--- Evaluating Regression Model ---")
y_pred = model_regression.predict(X_test_tensor).flatten() # Flatten to 1D array

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # RMSE is the square root of MSE
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# --- 5. Reflect on Metric Choices ---
print("\n--- Reflection on Regression Metric Choices ---")
print("1. **MAE vs. MSE:**")
print("   - **MAE (Mean Absolute Error):** Sum of absolute differences between predictions and actual values. It's robust to outliers because it treats all errors linearly.")
print("   - **MSE (Mean Squared Error):** Sum of squared differences. It penalizes larger errors more heavily due to the squaring, making it more sensitive to outliers. Useful when large errors are particularly undesirable.")
print("   - **Preference:** Choose MAE when you want errors to be directly proportional to the magnitude of the error and you need robustness to outliers. Choose MSE when you want to strongly penalize larger errors.")

print("\n2. **RMSE vs. MSE:**")
print("   - **RMSE (Root Mean Squared Error):** The square root of MSE. It's often preferred over MSE because it has the same units as the target variable, making it easier to interpret. An RMSE of 10 means your average error is around 10 units of the target variable.")

print("\n3. **R-squared (R²):**")
print("   - **What it represents:** The proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where 1 indicates that the model explains all the variability of the response data around its mean. A value of 0 indicates that the model explains no variability.")
print("   - **Limitations:** R² can sometimes increase just by adding more independent variables, even if they don't significantly improve the model's predictive power (adjusted R² addresses this but isn't included here). It doesn't tell you if your chosen features are the best or if the model is biased. A high R² doesn't guarantee a good model, nor does a low R² imply a bad one if the inherent noise in the data is high.")