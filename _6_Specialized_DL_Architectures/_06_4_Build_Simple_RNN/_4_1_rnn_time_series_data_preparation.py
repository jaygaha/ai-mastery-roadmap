"""
Time Series Data Preparation for RNNs

This script demonstrates how to prepare univariate time series data for training
a Recurrent Neural Network. The key concept is creating "sliding windows" where
each window of past observations becomes an input sample, and the next value
becomes the target to predict.

Example with look_back=3:
    Original series: [10, 20, 30, 40, 50, 60]
    
    Sample 1: Input=[10, 20, 30] → Target=40
    Sample 2: Input=[20, 30, 40] → Target=50
    Sample 3: Input=[30, 40, 50] → Target=60

Key concepts covered:
    - Creating sliding window sequences from time series data
    - Normalizing data with MinMaxScaler (important for neural networks)
    - Reshaping data to match RNN input requirements: (samples, timesteps, features)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_time_series_dataset(data, look_back=1):
    """
    Transform a univariate time series into supervised learning sequences.
    
    Think of it like this: if you want to predict tomorrow's temperature,
    you might look at the last 'look_back' days of temperatures (the input)
    to predict the next day (the output).
    
    Args:
        data (np.array): The input time series with shape (n_samples, 1).
                         Each row is one observation.
        look_back (int): The number of previous time steps to use as input features.
                         Also called the "window size" or "sequence length".
    
    Returns:
        tuple: (X, y) where:
            - X: Input sequences with shape (n_samples - look_back, look_back)
            - y: Target values with shape (n_samples - look_back,)
    
    Example:
        >>> data = np.array([[1], [2], [3], [4], [5]])
        >>> X, y = create_time_series_dataset(data, look_back=2)
        >>> print(X)  # [[1, 2], [2, 3], [3, 4]]
        >>> print(y)  # [3, 4, 5]
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        # Extract a window of 'look_back' values as input
        input_sequence = data[i:(i + look_back), 0]
        X.append(input_sequence)
        # The next value after the window is our target
        target_value = data[i + look_back, 0]
        y.append(target_value)
    return np.array(X), np.array(y)


# =============================================================================
# Generate Example Time Series Data
# =============================================================================
# We create a synthetic sine wave with some random noise added.
# This mimics real-world data that has both patterns and unpredictability.
time_steps = 100
series = np.sin(np.linspace(0, 20, time_steps)) + np.random.normal(0, 0.1, time_steps)

# Reshape to (timesteps, 1) - required format for MinMaxScaler
# The scaler expects 2D input: (samples, features)
series = series.reshape(-1, 1)

# =============================================================================
# Normalize the Data
# =============================================================================
# Why normalize? Neural networks train better when inputs are in a small range.
# MinMaxScaler transforms values to [0, 1], which:
#   1. Helps with gradient-based optimization
#   2. Prevents features with large values from dominating
#   3. Makes the tanh activation function in RNNs more effective
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_series = scaler.fit_transform(series)

# =============================================================================
# Create Training Sequences
# =============================================================================
# look_back=10 means: use 10 past observations to predict the next one
look_back = 10
X_ts, y_ts = create_time_series_dataset(scaled_series, look_back)

# Reshape input for RNN: [samples, time_steps, features]
# - samples: number of training examples (90 in this case: 100 - 10)
# - time_steps: length of each sequence (10, our look_back value)
# - features: dimensions per timestep (1 for univariate time series)
X_ts = np.reshape(X_ts, (X_ts.shape[0], X_ts.shape[1], 1))

# =============================================================================
# Verify Output Shapes
# =============================================================================
print("=" * 50)
print("Time Series Data Preparation Complete!")
print("=" * 50)
print(f"Original series length: {time_steps}")
print(f"Look-back window size: {look_back}")
print(f"Number of samples created: {X_ts.shape[0]}")
print()
print(f"X_ts shape: {X_ts.shape}")
print(f"  → {X_ts.shape[0]} samples, {X_ts.shape[1]} timesteps, {X_ts.shape[2]} feature")
print(f"y_ts shape: {y_ts.shape}")
print(f"  → {y_ts.shape[0]} target values to predict")