"""
Exercise 1: Input Shape Experiment

Goal: Understand how 'input_shape' must match the data dimensions.

1.  Run the code below. It will crash because the model expects 2 numbers per input (input_shape=[2]),
    but we are feeding it 1 number per input (xs is 1D).
2.  Fix the crash by changing 'xs' to be 2-dimensional.
    Example: xs = np.array([[-1.0, 0.5], [0.0, 1.0], ...])
3.  Update the prediction input to also match this shape.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define model expecting 2 inputs
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[2])
])

model.compile(optimizer='adam', loss='mean_squared_error')

# --- ERROR EXPECTED HERE ---
# Original data (1D) which causes mismatch
# xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

# FIXED DATA (2D) - Uncomment to fix
xs = np.array([[-1.0, 0.5], [0.0, 1.0], [1.0, 1.5], [2.0, 2.0], [3.0, 2.5], [4.0, 3.0]], dtype=float)

# Arbitrary target data
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train
model.fit(xs, ys, epochs=500)

# Make Prediction (Must also be 2D array)
# We pass [[10.0, 5.0]] as input
print(f"Prediction: {model.predict(np.array([[10.0, 5.0]]))}")