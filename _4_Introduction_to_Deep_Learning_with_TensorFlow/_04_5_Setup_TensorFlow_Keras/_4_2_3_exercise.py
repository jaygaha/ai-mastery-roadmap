"""
Exercise 3: Explore Different Optimizers and Loss Functions

Goal: Change how the model "learns" and "measures error".

1.  Optimizer: Change from 'adam' to 'sgd' (Stochastic Gradient Descent).
    - SGD is the classic way to train, but sometimes slower or less stable than Adam.
2.  Loss: Change from 'mean_squared_error' to 'mae' (Mean Absolute Error).
    - MAE measures the absolute difference, not squared.
3.  Train and compare results.
    - Does it converge faster?
    - Is the final prediction different?
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Use the simple linear model again (units=1, no activation or linear activation)
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1])
])

# EXPERIMENT HERE:
# Change optimizer to 'sgd'
# Change loss to 'mae'
model.compile(optimizer='sgd', loss='mae')

# Data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train
model.fit(xs, ys, epochs=500)

# Prediction
print(f"Prediction for 10.0: {model.predict(np.array([10.0]))}")
# Compare this result to the original 'adam' + 'mean_squared_error' result (approx 19.0).
# SGD might need more epochs or a different learning rate to get as close as Adam did in the same time.