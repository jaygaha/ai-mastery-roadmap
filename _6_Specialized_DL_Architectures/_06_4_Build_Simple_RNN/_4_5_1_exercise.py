"""
Exercise 1 Solution: Time Series Forecasting with Increased Look-Back

This exercise explores how increasing the look-back window affects RNN performance.

Objective:
    Modify the time series forecasting example to use look_back=20 instead of 10.
    Observe how training loss changes and visualize predicted vs. actual values.

Key learning points:
    - Larger look_back = more historical context for predictions
    - Trade-off: More context vs. fewer training samples
    - Larger windows may help with longer patterns but can be harder to train

Expected observations:
    - With look_back=20, the model sees more of the sine wave pattern
    - This might improve predictions if the pattern spans that length
    - However, we also get fewer training samples (180 vs. 190)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# =============================================================================
# 1. Generate Synthetic Time Series Data
# =============================================================================
def generate_time_series(num_points, freq=0.05, amplitude=1.0, noise_level=0.1):
    """Generate a noisy sine wave for experimentation."""
    time = np.arange(num_points)
    series = amplitude * np.sin(time * freq) + np.random.randn(num_points) * noise_level
    return series


data = generate_time_series(200)


# =============================================================================
# 2. Preprocess Data: Scaling
# =============================================================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))


# =============================================================================
# 3. Create Sequences for RNN Input
# =============================================================================
def create_sequences(data, look_back):
    """Create sliding window sequences for supervised learning."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)


# ========================== EXERCISE MODIFICATION ============================
# Changed from look_back=10 to look_back=20
# This means the model now sees 20 past values instead of 10
look_back = 20
# =============================================================================

X, y = create_sequences(scaled_data, look_back)

# Reshape X for RNN: (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)
print(f"With look_back={look_back}:")
print(f"  Number of training samples: {X.shape[0]}")
print(f"  Sequence length: {X.shape[1]}")


# =============================================================================
# 4. Split Data into Training and Testing Sets
# =============================================================================
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# =============================================================================
# 5. Build the Simple RNN Model
# =============================================================================
model_ts = Sequential([
    tf.keras.Input(shape=(look_back, 1)),
    SimpleRNN(units=50, activation='relu'),
    Dense(units=1)
])

model_ts.compile(optimizer='adam', loss='mean_squared_error')


# =============================================================================
# 6. Train the Model
# =============================================================================
# Capture training history to analyze loss progression
history = model_ts.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Compare with look_back=10: The loss trajectory may differ
print(f"\nFinal Training Loss (look_back={look_back}): {history.history['loss'][-1]:.4f}")
print("(Compare this with look_back=10 to see the difference)")


# =============================================================================
# 7. Make Predictions
# =============================================================================
train_predict_scaled = model_ts.predict(X_train)
test_predict_scaled = model_ts.predict(X_test)


# =============================================================================
# 8. Inverse Transform Predictions to Original Scale
# =============================================================================
train_predict = scaler.inverse_transform(train_predict_scaled)
test_predict = scaler.inverse_transform(test_predict_scaled)
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))


# =============================================================================
# 9. Evaluate Model Performance
# =============================================================================
train_rmse = np.sqrt(mean_squared_error(y_train_original, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_original, test_predict))

print(f"\nModel Performance:")
print(f"  Train RMSE: {train_rmse:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")


# ========================== EXERCISE ADDITION ================================
# Visualization: Plotting Predicted vs. Actual Values for the Test Set
# =============================================================================

# Plot 1: Full test set comparison
plt.figure(figsize=(15, 6))
plt.plot(y_test_original, label='Actual Test Values', color='blue', alpha=0.7)
plt.plot(test_predict, label='Predicted Test Values', color='red', linestyle='--')
plt.title(f'Time Series Forecasting: Actual vs. Predicted (Test Set, look_back={look_back})')
plt.xlabel('Time Step (relative to test set start)')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 2: First 50 points for detailed view (as requested: "portion of the test set")
# This zoomed view makes it easier to see how closely predictions match actual values
num_points_to_show = min(50, len(y_test_original))
plt.figure(figsize=(15, 6))
plt.plot(y_test_original[:num_points_to_show], label='Actual Test Values (First 50)', 
         color='blue', marker='o', markersize=4, alpha=0.7)
plt.plot(test_predict[:num_points_to_show], label='Predicted Test Values (First 50)', 
         color='red', marker='x', markersize=4, linestyle='--')
plt.title(f'Time Series Forecasting: Detailed View (First {num_points_to_show} Test Points, look_back={look_back})')
plt.xlabel('Time Step (relative to test set start)')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# =============================================================================