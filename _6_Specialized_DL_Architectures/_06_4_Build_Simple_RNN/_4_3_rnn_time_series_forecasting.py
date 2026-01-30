"""
Simple RNN for Time Series Forecasting

This script demonstrates the complete workflow for building and training
a Simple RNN to predict future values in a time series.

Workflow:
    1. Generate synthetic time series data (sine wave with noise)
    2. Normalize data to [0, 1] range (helps RNN training)
    3. Create sliding window sequences (supervised learning format)
    4. Split into training and testing sets
    5. Build and train the SimpleRNN model
    6. Evaluate performance and visualize predictions

Key concepts:
    - Why scaling matters for RNNs
    - How to structure time series for supervised learning
    - Inverse transforming predictions back to original scale
    - Visualizing model performance over time
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# =============================================================================
# 1. Generate Synthetic Time Series Data
# =============================================================================
def generate_time_series(num_points, freq=0.05, amplitude=1.0, noise_level=0.1):
    """
    Generate a sine wave with added random noise.
    
    This simulates real-world time series data which typically has:
    - An underlying pattern (the sine wave)
    - Random fluctuations (the noise)
    
    Args:
        num_points: Number of data points to generate
        freq: Frequency of the sine wave (higher = more oscillations)
        amplitude: Height of the wave peaks
        noise_level: Standard deviation of random noise
    
    Returns:
        np.array: The generated time series
    """
    time = np.arange(num_points)
    series = amplitude * np.sin(time * freq) + np.random.randn(num_points) * noise_level
    return series


# Generate 200 data points
data = generate_time_series(200)
print(f"Generated time series with {len(data)} points")


# =============================================================================
# 2. Preprocess Data: Scaling
# =============================================================================
# Why scale? RNNs use tanh activation which outputs [-1, 1].
# If input values are very large, the network can struggle to learn.
# MinMaxScaler normalizes to [0, 1], making training more stable.

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))
print(f"Data scaled to range [{scaled_data.min():.2f}, {scaled_data.max():.2f}]")


# =============================================================================
# 3. Create Sequences for RNN Input
# =============================================================================
def create_sequences(data, look_back):
    """
    Transform time series into supervised learning format.
    
    For each position, we create:
    - X: A window of 'look_back' past values
    - y: The next value to predict
    
    Args:
        data: Scaled time series with shape (n, 1)
        look_back: How many past observations to use as features
    
    Returns:
        X, y: Input sequences and corresponding targets
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)


look_back = 10  # Use 10 past observations to predict the next one
X, y = create_sequences(scaled_data, look_back)

# Reshape X for RNN: (samples, timesteps, features)
# - samples: how many examples we have
# - timesteps: our look_back window size
# - features: 1 (univariate time series)
X = X.reshape(X.shape[0], X.shape[1], 1)
print(f"Created {X.shape[0]} sequences of length {look_back}")


# =============================================================================
# 4. Split Data into Training and Testing Sets
# =============================================================================
# Important: For time series, we split chronologically (not randomly!)
# We train on earlier data and test on later data.
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")


# =============================================================================
# 5. Build the Simple RNN Model
# =============================================================================
# Model architecture:
# - Input: sequences of shape (look_back, 1)
# - SimpleRNN: Processes the sequence, outputs a 50-dim hidden state
# - Dense: Maps the hidden state to a single prediction

model = Sequential([
    tf.keras.Input(shape=(look_back, 1)),
    SimpleRNN(units=50, activation='relu'),  # 50 units = size of hidden state
    Dense(units=1)  # Output: single value prediction
])

# Compile with MSE loss (standard for regression tasks)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# =============================================================================
# 6. Train the Model
# =============================================================================
print("\nTraining the model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)


# =============================================================================
# 7. Make Predictions
# =============================================================================
train_predict_scaled = model.predict(X_train)
test_predict_scaled = model.predict(X_test)


# =============================================================================
# 8. Inverse Transform Predictions to Original Scale
# =============================================================================
# We trained on scaled data, so predictions are also scaled.
# To interpret results, we need to convert back to original units.
train_predict = scaler.inverse_transform(train_predict_scaled)
test_predict = scaler.inverse_transform(test_predict_scaled)
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))


# =============================================================================
# 9. Evaluate Model Performance
# =============================================================================
# RMSE (Root Mean Squared Error) measures average prediction error
# in the same units as the original data.
train_rmse = np.sqrt(mean_squared_error(y_train_original, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_original, test_predict))

print("\n" + "=" * 50)
print("Model Performance")
print("=" * 50)
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print("\nNote: Lower RMSE = better predictions")


# =============================================================================
# 10. Visualize Results
# =============================================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(data, label='Original Data', alpha=0.7)

# Align train predictions with the correct time indices
train_plot = np.empty_like(data)
train_plot[:] = np.nan
train_plot[look_back:len(train_predict) + look_back] = train_predict.flatten()
plt.plot(train_plot, label='Train Predictions')

# Align test predictions with the correct time indices
test_plot = np.empty_like(data)
test_plot[:] = np.nan
test_plot[len(train_predict) + look_back:len(data)] = test_predict.flatten()
plt.plot(test_plot, label='Test Predictions')

plt.title('Time Series Forecasting with Simple RNN')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()