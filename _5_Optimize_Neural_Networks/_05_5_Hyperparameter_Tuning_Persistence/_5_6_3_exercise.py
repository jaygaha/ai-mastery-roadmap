"""
Exercise 3: Model Selection & Saving - The Final Step

Goal: Practice the workflow of training, saving, and then reloading a model for use.

Tasks:
a) Train a simple model on dummy data.
b) Save the entire model (SavedModel format).
c) Load it back and make a "production" prediction.
d) Save only the weights and load them into a fresh model with the same architecture.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, TFSMLayer
from tensorflow.keras.optimizers import Adam
import numpy as np
import shutil
import os

def run_exercise_3():
    print("--- Step a: Train the model on dummy data ---")
    input_dim = 10
    # Create a simple churn model
    model = Sequential([
        Input(shape=(input_dim,)), 
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Dummy data
    X_train = np.random.rand(100, input_dim)
    y_train = np.random.randint(0, 2, 100)

    # Train
    model.fit(X_train, y_train, epochs=5, verbose=0)
    print("Model trained successfully.")

    print("\n--- Step b: Save the entire trained model (SavedModel) ---")
    saved_model_path = 'my_advanced_churn_model'
    # cleanup if exists
    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)
        
    model.export(saved_model_path) # Use export for SavedModel format in Keras 3 / recent TF
    # or model.save(saved_model_path) depending on version, but export is safer for pure SavedModel artifact
    print(f"Model saved to {saved_model_path}")

    print("\n--- Step c: Load the model from the directory ---")
    # For inference-only loading of SavedModel in recent Keras, we use TFSMLayer or keras.models.load_model
    # Note: load_model on a SavedModel directory might return a high-level object or a layer depending on TF/Keras version.
    # Let's use the standard load_model first.
    try:
        loaded_model = load_model(saved_model_path)
        print("Model loaded using load_model.")
    except Exception as e:
        print(f"load_model failed (might be Keras 3 vs TF behavior): {e}")
        print("Trying TFSMLayer...")
        loaded_model = TFSMLayer(saved_model_path, call_endpoint='serving_default')

    print("\n--- Step d: Print summary ---")
    if hasattr(loaded_model, 'summary'):
        loaded_model.summary()
    else:
        print("Loaded object is likely a TFSMLayer or inference function which doesn't support .summary() directly.")

    print("\n--- Step e: Make a prediction ---")
    dummy_input = np.random.rand(1, input_dim)
    prediction = loaded_model(dummy_input)
    # Output might be a dict for TFSMLayer
    if isinstance(prediction, dict):
        print(f"Prediction: {prediction['output_0'].numpy()}")
    else:
        print(f"Prediction: {prediction}")


    print("\n--- Step f: Save only the weights to .h5 ---")
    weights_path = 'my_advanced_churn_weights.weights.h5'
    if os.path.exists(weights_path):
        os.remove(weights_path)
        
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")

    print("\n--- Step g: Re-define model and load weights ---")
    new_model = Sequential([
        Input(shape=(input_dim,)), 
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # Architecture must match exactly
    new_model.load_weights(weights_path)
    print("Weights loaded into new model.")

    print("\n--- Step h: Verify with prediction ---")
    pred_new = new_model.predict(dummy_input)
    print(f"Prediction from reloaded weights model: {pred_new}")

if __name__ == "__main__":
    run_exercise_3()