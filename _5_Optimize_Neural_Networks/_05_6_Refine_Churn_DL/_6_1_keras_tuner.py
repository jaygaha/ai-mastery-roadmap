"""
Keras Tuner: Introduction to Automated Hyperparameter Tuning

This script demonstrates how to use Keras Tuner to automatically find the best
hyperparameters for a neural network. Instead of manually trying different
combinations of layers, neurons, and learning rates, Keras Tuner does the
searching for you!

Key Concepts Covered:
    - Defining a tunable model with hp.Int(), hp.Float(), hp.Boolean(), hp.Choice()
    - Using RandomSearch tuner to explore hyperparameter space
    - Setting up EarlyStopping callback during tuning
    - Retrieving and evaluating the best model

What This Script Does:
    1. Creates dummy churn prediction data (or loads real data if available)
    2. Defines a model-building function with tunable hyperparameters
    3. Uses RandomSearch to find the best configuration
    4. Evaluates the best model on test data

Run with: conda run -n tf_env python _6_1_keras_tuner.py
"""

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load and preprocess the churn data (assuming it's already prepared)
# For demonstration, let's create dummy data similar to the churn case study
np.random.seed(42)
num_samples = 1000
features = np.random.rand(num_samples, 10) # 10 features
target = np.random.randint(0, 2, num_samples) # Binary target (churn/no churn)

# Simulate some categorical features
gender = np.random.choice(['Male', 'Female'], num_samples)
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples)
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], num_samples)

df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(10)])
df['gender'] = gender
df['contract'] = contract
df['internet_service'] = internet_service
df['churn'] = target

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['gender', 'contract', 'internet_service'], drop_first=True)

X = df.drop('churn', axis=1)
y = df['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features (assuming features are numerical)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame if feature names are needed later (optional for Keras)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)


def build_model(hp):
    model = keras.Sequential()
    
    # Define the input shape explicitly (Modern Keras 3 way)
    model.add(keras.Input(shape=(X_train_scaled.shape[1],)))
    
    # First hidden layer
    model.add(keras.layers.Dense(
        units=hp.Int('units_1', min_value=32, max_value=256, step=32), # Tune number of units
        activation='relu'
    ))
    
    # Add optional second hidden layer
    if hp.Boolean("has_second_layer"):
        model.add(keras.layers.Dense(
            units=hp.Int('units_2', min_value=32, max_value=256, step=32),
            activation='relu'
        ))
        
    # Add optional dropout layer
    if hp.Boolean("has_dropout"):
        model.add(keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))

    # Output layer
    model.add(keras.layers.Dense(1, activation='sigmoid')) # Binary classification for churn

    # Tune the learning rate for the Adam optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Instantiate the tuner (e.g., RandomSearch)
# We can also use Hyperband or BayesianOptimization here
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy', # Metric to optimize for
    max_trials=10,             # Number of different models to try
    executions_per_trial=1,    # Number of models to train for each trial (for robustness)
    directory='churn_hp_tuning', # Directory to store results
    project_name='churn_prediction'
)

# Summarize the search space
tuner.search_space_summary()

# Start the hyperparameter search
# Use callbacks for early stopping during tuning to save time
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train_scaled, y_train, 
             epochs=50, 
             validation_split=0.2, 
             callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units in the first hidden layer is {best_hps.get('units_1')}.
The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
Whether to include a second layer: {best_hps.get('has_second_layer')}.
If a second layer is included, its units are: {best_hps.get('units_2') if best_hps.get('has_second_layer') else 'N/A'}.
Whether to include dropout: {best_hps.get('has_dropout')}.
If dropout is included, its rate is: {best_hps.get('dropout_rate') if best_hps.get('has_dropout') else 'N/A'}.
""")

# Build the best model and evaluate it
best_model = tuner.get_best_models(num_models=1)[0]
loss, accuracy = best_model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")