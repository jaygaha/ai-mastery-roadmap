"""
Exercise 2: Activation Functions and Units

Goal: See how adding neurons and activation functions changes the model's behavior.

1.  Original model: units=1, activation=None (Linear).
    - Prediction is just a straight line.
2.  Modified model below: units=1, activation='relu'.
    - ReLU (Rectified Linear Unit) outputs 0 for negative inputs.
3.  Train it and predict for 10.0 and -5.0.
    - Notice valid positive predictions.
    - Notice that negative inputs result in 0 (or close to it) because of ReLU.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define model with ReLU activation
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1], activation='relu')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train
model.fit(xs, ys, epochs=500)

# Predictions
print(f"Prediction for 10.0: {model.predict(np.array([10.0]))}")
print(f"Prediction for -5.0: {model.predict(np.array([-5.0]))}")
# Observation: For 10.0, it should be close to 19.
# For -5.0, it will be 0.0 because ReLU turns all negative inputs to 0.