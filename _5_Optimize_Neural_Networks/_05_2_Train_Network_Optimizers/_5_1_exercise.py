"""
Exercise 1: Understanding Underfitting vs. Overfitting

This exercise demonstrates the impact of training for too few or too many epochs.
We'll compare a model trained for 1 epoch (likely underfitted) with one trained
for 100 epochs (potentially overfitted).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
print("EXERCISE 1: Underfitting vs. Overfitting")
print("=" * 70)

# Model with 1 epoch (likely underfitted)
print("\n--- Training Model with 1 Epoch ---")
model_underfit = create_model()
model_underfit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_underfit = model_underfit.fit(
    X_train_scaled, y_train,
    epochs=1,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Evaluate
val_acc_underfit = history_underfit.history['val_accuracy'][-1]
loss_underfit, test_acc_underfit = model_underfit.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\n1 Epoch Model Results:")
print(f"  Validation Accuracy: {val_acc_underfit:.4f}")
print(f"  Test Accuracy: {test_acc_underfit:.4f}")

# Model with 100 epochs (potentially overfitted)
print("\n--- Training Model with 100 Epochs ---")
model_overfit = create_model()
model_overfit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_overfit = model_overfit.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    verbose=0  # Set to 0 to avoid too much output
)

# Evaluate
val_acc_overfit = history_overfit.history['val_accuracy'][-1]
loss_overfit, test_acc_overfit = model_overfit.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\n100 Epoch Model Results:")
print(f"  Validation Accuracy: {val_acc_overfit:.4f}")
print(f"  Test Accuracy: {test_acc_overfit:.4f}")

# Analysis
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print("\n1. Which model shows signs of underfitting?")
if test_acc_underfit < 0.7:  # Arbitrary threshold for demonstration
    print(f"   → The 1-epoch model (test accuracy: {test_acc_underfit:.4f})")
    print("   → It hasn't trained long enough to learn the patterns in the data")
else:
    print(f"   → Neither model is severely underfitted")

print("\n2. Which model might be overfitting?")
gap_underfit = abs(val_acc_underfit - test_acc_underfit)
gap_overfit = abs(val_acc_overfit - test_acc_overfit)

print(f"   → 1-epoch model gap (val vs test): {gap_underfit:.4f}")
print(f"   → 100-epoch model gap (val vs test): {gap_overfit:.4f}")

if gap_overfit > gap_underfit * 1.5:
    print(f"   → The 100-epoch model shows more signs of overfitting")
    print(f"   → Larger gap between validation and test accuracy suggests")
    print(f"      the model memorized training data rather than learning patterns")
else:
    print(f"   → Both models generalize reasonably well")

print("\n3. What's the sweet spot?")
print(f"   → For this dataset, somewhere between 1 and 100 epochs")
print(f"   → Monitor validation accuracy during training and stop when it plateaus")
print(f"   → Typical range: 10-30 epochs for this type of problem")

print("\n" + "=" * 70)
print("KEY TAKEAWAY")
print("=" * 70)
print("Too few epochs → underfitting (poor performance)")
print("Too many epochs → overfitting (good training, poor test performance)")
print("Sweet spot → balance between learning and generalizing")
print("=" * 70)
