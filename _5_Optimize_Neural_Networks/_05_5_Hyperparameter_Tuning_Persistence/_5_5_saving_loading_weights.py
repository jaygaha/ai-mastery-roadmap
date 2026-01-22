import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

"""
STEP 1: Save only weights
Setup: Define and Train a dummy model
"""
input_dim = 10
# Create a dummy model (similar to churn prediction)
model = Sequential([
    Input(shape=(input_dim,)), 
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Dummy data for training
X_train_dummy = np.random.rand(100, 10)
y_train_dummy = np.random.randint(0, 2, 100)

# Train the model
model.fit(X_train_dummy, y_train_dummy, epochs=5, batch_size=32, verbose=0)
print("Model trained.")


"""
Save ONLY Weights
"""
# Save only the weights
# check if file exists and remove it
if os.path.exists('my_churn_model_weights.weights.h5'):
    os.remove('my_churn_model_weights.weights.h5')

model.save_weights('my_churn_model_weights.weights.h5')
print("Model weights saved to my_churn_model_weights.weights.h5")


"""
STEP 2: Load ONLY Weights
"""
# To load weights, you need a model with the SAME architecture
loaded_model_weights = Sequential([
    Input(shape=(input_dim,)), 
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Load the weights into the new model
loaded_model_weights.load_weights('my_churn_model_weights.weights.h5')
print("Model weights loaded successfully.")

# Verify with a prediction
dummy_input = np.random.rand(1, 10)
prediction = loaded_model_weights.predict(dummy_input)
print(f"Prediction using loaded weights: {prediction}")
