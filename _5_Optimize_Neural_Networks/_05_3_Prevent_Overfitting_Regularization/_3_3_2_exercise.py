"""
Exercise

2. Dropout Experiment 
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Data (Reusing from Exercise 1)
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = np.sin(X * 2) + np.random.normal(0, 0.2, X.shape)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define a Complex Model ---
# We use a function here because we want to create two IDENTICAL starting architectures.
# This ensures the only difference is the Dropout layer.
def build_complex_architecture():
    # A model deep enough to overfit easily on this small dataset
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)), 
        tf.keras.layers.Dense(128, activation='relu'),
        # We will insert Dropout here in the second model
        tf.keras.layers.Dense(64, activation='relu'), 
        # And here
        tf.keras.layers.Dense(1)
    ])
    return model

# 2. Train Two Models

# --- Model C: No Dropout (The "Overfitter") ---
print("Training Model C (No Dropout)...")
model_C = build_complex_architecture()
model_C.compile(optimizer='adam', loss='mse')
# We train for 150 epochs to give it plenty of time to memorize the noise.
history_C = model_C.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val), verbose=0) 


# --- Model D: With Dropout (The "Robust One") ---
print("Training Model D (With Dropout)...")
dropout_rate = 0.3 # 30% of neurons are dropped each pass!
model_D = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dropout(dropout_rate), # BLOCK 1
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate), # BLOCK 2
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate), # BLOCK 3
    
    tf.keras.layers.Dense(1)
])
model_D.compile(optimizer='adam', loss='mse')
history_D = model_D.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# 3. Visualize Results

# Generate predictions for plotting
X_test_plot = np.linspace(-4, 4, 300).reshape(-1, 1)
y_pred_C = model_C.predict(X_test_plot)
y_pred_D = model_D.predict(X_test_plot)

plt.figure(figsize=(15, 6))

# Plot 1: Model Predictions
plt.subplot(1, 2, 1)
plt.scatter(X, y, label='Data (with noise)', alpha=0.6, s=10)
plt.plot(X_test_plot, y_pred_C, color='red', label='Model C (No Dropout)', linewidth=2)
plt.plot(X_test_plot, y_pred_D, color='blue', label=f'Model D (Dropout {dropout_rate})', linewidth=2)
plt.title('Prediction Comparison')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss Curves
plt.subplot(1, 2, 2)
plt.plot(history_C.history['val_loss'], label='Model C Val Loss (No Dropout)', linestyle='--', color='red')
plt.plot(history_D.history['val_loss'], label='Model D Val Loss (With Dropout)', linestyle='--', color='blue')
plt.title('Validation Loss Comparison (Lower is Better)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Analysis
print("\n--- Analysis ---")
print("1. Model C (Red Line): Does it look jagged? This means it's over-reacting to noise.")
print("2. Model D (Blue Line): Does it look smoother? Dropout forces it to learn the 'big picture'.")
print("3. Check the Validation Loss graph. Model C's loss might start increasing (getting worse) after a while as it overfits.")
print("   Model D should remain lower and more stable.")
