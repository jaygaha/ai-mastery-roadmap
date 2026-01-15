"""
Exercises and Practice Activities

1. Modify Hidden Layers and Neurons (Churn Prediction): Take the provided model definition for the binary classification (customer churn) problem. Experiment with the number of hidden layers and the number of neurons in each layer.
    - Create an MLP with three hidden layers (e.g., 128, 64, 32 units).
    - Create an MLP with a single hidden layer (e.g., 100 units).
    - Compare the training and validation accuracy/loss of these different architectures. What observations can you make about the impact of network depth and width?

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the preprocessed data (assuming you have a CSV from previous steps)
# Replace 'your_preprocessed_churn_data.csv' with your actual file path
try:
    df = pd.read_csv('../../_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling/Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Error: 'your_preprocessed_churn_data.csv' not found.")
    print("Please ensure your preprocessed churn data is saved and correctly referenced.")
    print("For example, you might need to run the data preprocessing steps from Module 2 and save the output.")
    exit()

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

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features (important for NNs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to TensorFlow tensors
X_train_tensor = tf.constant(X_train_scaled, dtype=tf.float32)
X_test_tensor = tf.constant(X_test_scaled, dtype=tf.float32)
y_train_tensor = tf.constant(y_train.values, dtype=tf.float32)
y_test_tensor = tf.constant(y_test.values, dtype=tf.float32)

input_dim = X_train_scaled.shape[1]

# --- 1. Baseline Model (1 hidden layer, 32 neurons) ---
print("\n--- Training Baseline Model (1 Hidden Layer, 32 Neurons) ---")
model_baseline = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation='relu', name='hidden_layer_1'),
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

model_baseline.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

history_baseline = model_baseline.fit(X_train_tensor, y_train_tensor,
                                      epochs=50,
                                      batch_size=32,
                                      validation_split=0.1, # Use a small validation split during training
                                      verbose=0) # Set verbose to 1 to see progress

y_pred_probs_baseline = model_baseline.predict(X_test_tensor)
y_pred_baseline = (y_pred_probs_baseline > 0.5).astype(int)

print(f"Baseline Model Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"Baseline Model Precision: {precision_score(y_test, y_pred_baseline):.4f}")
print(f"Baseline Model Recall: {recall_score(y_test, y_pred_baseline):.4f}")
print(f"Baseline Model F1-Score: {f1_score(y_test, y_pred_baseline):.4f}")

# --- 2. Model with Two Hidden Layers (64, 32 neurons) ---
print("\n--- Training Model with Two Hidden Layers (64, 32 Neurons) ---")
model_two_layers = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu', name='hidden_layer_1'),
    layers.Dense(32, activation='relu', name='hidden_layer_2'),
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

model_two_layers.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

history_two_layers = model_two_layers.fit(X_train_tensor, y_train_tensor,
                                          epochs=50,
                                          batch_size=32,
                                          validation_split=0.1,
                                          verbose=0)

y_pred_probs_two_layers = model_two_layers.predict(X_test_tensor)
y_pred_two_layers = (y_pred_probs_two_layers > 0.5).astype(int)

print(f"Two Layers Model Accuracy: {accuracy_score(y_test, y_pred_two_layers):.4f}")
print(f"Two Layers Model Precision: {precision_score(y_test, y_pred_two_layers):.4f}")
print(f"Two Layers Model Recall: {recall_score(y_test, y_pred_two_layers):.4f}")
print(f"Two Layers Model F1-Score: {f1_score(y_test, y_pred_two_layers):.4f}")

# --- 3. Model with More Neurons per Layer (e.g., 1 hidden layer, 128 neurons) ---
print("\n--- Training Model with More Neurons (1 Hidden Layer, 128 Neurons) ---")
model_more_neurons = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu', name='hidden_layer_1'),
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

model_more_neurons.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

history_more_neurons = model_more_neurons.fit(X_train_tensor, y_train_tensor,
                                              epochs=50,
                                              batch_size=32,
                                              validation_split=0.1,
                                              verbose=0)

y_pred_probs_more_neurons = model_more_neurons.predict(X_test_tensor)
y_pred_more_neurons = (y_pred_probs_more_neurons > 0.5).astype(int)

print(f"More Neurons Model Accuracy: {accuracy_score(y_test, y_pred_more_neurons):.4f}")
print(f"More Neurons Model Precision: {precision_score(y_test, y_pred_more_neurons):.4f}")
print(f"More Neurons Model Recall: {recall_score(y_test, y_pred_more_neurons):.4f}")
print(f"More Neurons Model F1-Score: {f1_score(y_test, y_pred_more_neurons):.4f}")

print("\n--- Analysis of Results ---")
print("Compare the performance metrics (Accuracy, Precision, Recall, F1-Score) across the three models.")
print("Consider these questions:")
print("1. Did adding a second hidden layer significantly improve performance? Why or why not?")
print("2. Did increasing the number of neurons in a single layer improve performance? Why or why not?")
print("3. What are the potential trade-offs (e.g., training time, risk of overfitting) as model complexity increases?")
print("4. Observe the validation accuracy/loss from the `history` objects (e.g., `history_baseline.history['val_accuracy'][-1]`) to get an idea of overfitting during training.")
