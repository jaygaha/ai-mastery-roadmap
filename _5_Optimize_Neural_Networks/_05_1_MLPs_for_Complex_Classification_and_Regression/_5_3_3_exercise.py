"""
3. Implement Multi-Class Classification with Sparse Labels: Modify the multi-class classification example to use sparse_categorical_crossentropy 
    instead of categorical_crossentropy. This means the y_multi labels should not be one-hot encoded.

    - Remove the OneHotEncoder step for y_multi.
    - Adjust the loss function in model_multi.compile() accordingly.
    - Ensure the model still trains and evaluates correctly.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification # For generating synthetic multi-class data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# --- 1. Generate a Synthetic Multi-Class Dataset ---
print("--- Generating Synthetic Multi-Class Data ---")
n_samples = 1500
n_features = 20
n_classes = 4 # Example: 4 distinct classes (0, 1, 2, 3)

X, y = make_classification(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=10, # Features that are useful
                           n_redundant=5,    # Features that are combinations of others
                           n_classes=n_classes,
                           random_state=42,
                           n_clusters_per_class=1) # Makes classes more distinct initially

print(f"Generated X shape: {X.shape}")
print(f"Generated y shape: {y.shape}")
print(f"Number of classes: {np.unique(y).shape[0]}")
print(f"Sample y values (sparse labels): {y[:10]}")

# --- 2. Preprocess Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = tf.constant(X_train_scaled, dtype=tf.float32)
X_test_tensor = tf.constant(X_test_scaled, dtype=tf.float32)
y_train_tensor = tf.constant(y_train, dtype=tf.int32) # Keep y as integers for sparse_categorical_crossentropy
y_test_tensor = tf.constant(y_test, dtype=tf.int32)

input_dim = X_train_scaled.shape[1]
num_classes = np.unique(y_train).shape[0] # Get actual number of classes from data

# --- 3. Build and Train an MLP for Multi-Class Classification (Sparse Labels) ---
print("\n--- Building and Training Multi-Class MLP with Sparse Categorical Crossentropy ---")
model_multi_class = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu', name='hidden_layer_1'),
    layers.Dense(64, activation='relu', name='hidden_layer_2'),
    layers.Dense(num_classes, activation='softmax', name='output_layer_softmax') # Output neurons = num_classes, activation='softmax'
])

model_multi_class.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy', # Key for integer labels
                          metrics=['accuracy'])

print(model_multi_class.summary())

history_multi_class = model_multi_class.fit(X_train_tensor, y_train_tensor,
                                            epochs=70, # Increased epochs for multi-class
                                            batch_size=32,
                                            validation_split=0.1,
                                            verbose=1)

# --- 4. Evaluate the Model ---
print("\n--- Evaluating Multi-Class Model ---")
y_pred_probs = model_multi_class.predict(X_test_tensor)
y_pred_classes = np.argmax(y_pred_probs, axis=1) # Get the class with the highest probability

print(f"Multi-Class Model Accuracy: {accuracy_score(y_test, y_pred_classes):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=[f'Class {i}' for i in range(num_classes)]))

# --- 5. Reflect on Churn Prediction (Hypothetical Multi-Class) ---
print("\n--- Reflection on Hypothetical Multi-Class Churn Prediction ---")
print("If churn prediction were multi-class (e.g., 'no churn', 'churn to competitor A', 'churn to competitor B'):")
print(f"1. The output layer would need {num_classes} neurons (where num_classes is 4 in this example).")
print("2. The output layer's activation function would be 'softmax' to provide probabilities for each churn type.")
print("3. The loss function would change from 'binary_crossentropy' to 'sparse_categorical_crossentropy' if target labels are integers (0, 1, 2) or 'categorical_crossentropy' if target labels are one-hot encoded (e.g., [1,0,0], [0,1,0], [0,0,1]).")
print("4. `sparse_categorical_crossentropy` is useful because it directly accepts integer labels, saving the step of one-hot encoding your `y_train` and `y_test` arrays, which can simplify the data pipeline, especially when classes are numerous.")