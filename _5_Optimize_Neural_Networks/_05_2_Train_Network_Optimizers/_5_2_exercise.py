"""
Exercise 2: The Impact of Batch Size

This exercise demonstrates how batch size affects training dynamics, convergence,
and final performance. We'll compare a very small batch size (4) with a very
large batch size (512).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Generate dummy data for demonstration
np.random.seed(42)
num_samples = 10000
num_features = 10

# Create synthetic features
X = np.random.rand(num_samples, num_features) * 100

# Create synthetic target variable
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.rand(num_samples) * 10 > 70).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model creation function
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

print("=" * 70)
print("EXERCISE 2: Impact of Batch Size")
print("=" * 70)

# Model A: Very small batch size
print("\n--- Training Model A: Batch Size = 4 ---")
model_small_batch = create_model()
model_small_batch.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
history_small = model_small_batch.fit(
    X_train_scaled, y_train,
    epochs=15,
    batch_size=4,
    validation_split=0.1,
    verbose=1
)
time_small = time.time() - start_time

# Evaluate
loss_small, acc_small = model_small_batch.evaluate(X_test_scaled, y_test, verbose=0)

# Model B: Very large batch size
print("\n--- Training Model B: Batch Size = 512 ---")
model_large_batch = create_model()
model_large_batch.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
history_large = model_large_batch.fit(
    X_train_scaled, y_train,
    epochs=15,
    batch_size=512,
    validation_split=0.1,
    verbose=1
)
time_large = time.time() - start_time

# Evaluate
loss_large, acc_large = model_large_batch.evaluate(X_test_scaled, y_test, verbose=0)

# Analysis
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

# Calculate steps per epoch
train_size = int(len(X_train_scaled) * 0.9)  # 90% for training, 10% for validation
steps_small = train_size // 4
steps_large = train_size // 512

print("\n1. Steps per epoch:")
print(f"   → Batch size 4: {steps_small} steps per epoch")
print(f"   → Batch size 512: {steps_large} steps per epoch")
print(f"   → Small batch has {steps_small / steps_large:.1f}x more updates per epoch!")

print("\n2. Training dynamics (loss curve):")
# Calculate loss variance as a measure of "jumpiness"
loss_variance_small = np.var(history_small.history['loss'])
loss_variance_large = np.var(history_large.history['loss'])

print(f"   → Batch size 4 loss variance: {loss_variance_small:.6f}")
print(f"   → Batch size 512 loss variance: {loss_variance_large:.6f}")

if loss_variance_small > loss_variance_large:
    print(f"   → Small batch (4) is more erratic/jumpy")
    print(f"      (Higher variance = more noise in updates)")
else:
    print(f"   → Large batch (512) is smoother")

print("\n3. Training speed (wall-clock time):")
print(f"   → Batch size 4: {time_small:.2f} seconds")
print(f"   → Batch size 512: {time_large:.2f} seconds")

if time_large < time_small:
    print(f"   → Large batch is {time_small/time_large:.2f}x faster!")
    print(f"      (GPUs can process large batches more efficiently)")
else:
    print(f"   → Small batch is faster (unusual, may depend on hardware)")

print("\n4. Final test accuracy:")
print(f"   → Batch size 4: {acc_small:.4f}")
print(f"   → Batch size 512: {acc_large:.4f}")

if acc_small > acc_large:
    print(f"   → Small batch achieved better accuracy")
    print(f"      (More frequent updates helped find a better solution)")
elif acc_large > acc_small:
    print(f"   → Large batch achieved better accuracy")
    print(f"      (Smoother updates led to better convergence)")
else:
    print(f"   → Both achieved similar accuracy")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("Small batch sizes (4):")
print("  ✅ More updates per epoch → can learn faster")
print("  ✅ Noisier updates → can escape bad solutions")
print("  ❌ Slower wall-clock time → more computational overhead")
print("  ❌ Erratic training → harder to tune")
print()
print("Large batch sizes (512):")
print("  ✅ Faster wall-clock time → efficient GPU usage")
print("  ✅ Smoother training → more stable convergence")
print("  ❌ Fewer updates per epoch → may need more epochs")
print("  ❌ Risk of getting stuck → might miss better solutions")
print()
print("Typical choice: 32-128 for most problems (good balance)")
print("=" * 70)
