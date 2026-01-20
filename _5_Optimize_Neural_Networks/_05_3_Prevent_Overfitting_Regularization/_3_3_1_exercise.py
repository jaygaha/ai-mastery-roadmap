"""
Exercise

1. L2 Regularization Experiment
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Data
# We create a simple sine wave with added noise.
# The noise represents real-world imperfections that we don't want the model to memorize.
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = np.sin(X * 2) + np.random.normal(0, 0.2, X.shape)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Two Models

# --- Model A: No L2 Regularization ---
# This model is free to use large weights to fit every wiggly detail of the noise.
model_A = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_A.compile(optimizer='adam', loss='mse')
print("Training Model A (No Regularization)...")
history_A = model_A.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# --- Model B: With L2 Regularization ---
# This model is penalized for having large weights.
# The 'l2_lambda' controls how strong the penalty is.
l2_lambda = 0.01 
model_B = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,),
                          kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)), # Apply penalty
    tf.keras.layers.Dense(64, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)), # Apply penalty
    tf.keras.layers.Dense(1)
])
model_B.compile(optimizer='adam', loss='mse')
print(f"Training Model B (L2 Lambda={l2_lambda})...")
history_B = model_B.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# 3. Visualize Results

# Generate a smooth range of X values to plot the model's "curve"
X_test_plot = np.linspace(-4, 4, 300).reshape(-1, 1)
y_pred_A = model_A.predict(X_test_plot)
y_pred_B = model_B.predict(X_test_plot)

plt.figure(figsize=(15, 6))

# Plot 1: Model Predictions
plt.subplot(1, 2, 1)
plt.scatter(X, y, label='Original Noisy Data', alpha=0.6, s=10)
plt.plot(X_test_plot, y_pred_A, color='red', label='Model A (No L2) - Wiggly?', linewidth=2)
plt.plot(X_test_plot, y_pred_B, color='green', label=f'Model B (L2) - Smoother?', linewidth=2)
plt.title('Visual Comparison: Overfitting vs. Regularization')
plt.xlabel('Input X')
plt.ylabel('Output y')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss Curves
plt.subplot(1, 2, 2)
plt.plot(history_A.history['val_loss'], label='Model A Val Loss (No L2)', linestyle='--', color='red')
plt.plot(history_B.history['val_loss'], label='Model B Val Loss (With L2)', linestyle='--', color='green')
# We focus on Validation Loss because that tells us how well it generalizes.
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (Lower is Better)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Analysis
print("\n--- Analysis ---")
print("Check the plots:")
print("1. Did Model A try to fit the 'outliers' (extreme noise points)? This creates a jagged curve.")
print("2. Did Model B ignore the noise and draw a smoother curve through the middle?")
print(f"3. Look at the Validation Loss. Model B's loss should be lower or more stable than Model A's.")
print("\nTry changing 'l2_lambda' in the code to 0.1 or 0.0001 and run it again to see the effect!")