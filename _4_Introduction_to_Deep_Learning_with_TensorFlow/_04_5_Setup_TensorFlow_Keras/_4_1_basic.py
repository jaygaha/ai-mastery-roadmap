# STEP 1: Import the necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# STEP 2: Define the model
# We create a Sequential model (a linear stack of layers)
model = keras.Sequential([
    # Add a Dense layer with 1 unit (neuron)
    # input_shape=[1] means the model expects 1 number as input at a time
    layers.Dense(units=1, input_shape=[1])
])

# STEP 3: Compile the Model
# Configure the learning process
# optimizer='adam': The algorithm that minimizes the error
# loss='mean_squared_error': The method to calculate the error (MSE)
model.compile(optimizer='adam', loss='mean_squared_error')

# STEP 4: Prepare Data
# We want the model to learn the relationship: y = 2x - 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# STEP 5: Train the Model
# The model will try to "fit" the relationship between xs and ys
# It will loop through the data 500 times (epochs)
model.fit(xs, ys, epochs=500)

# STEP 6: Make Predictions
# Predict the value of y when x is 10.0
# Expected result: approx 19.0 (2 * 10 - 1)
print(model.predict(np.array([10.0])))