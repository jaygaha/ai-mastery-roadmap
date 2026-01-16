"""
Exercise 3: SGD with Momentum vs. Adam

This exercise demonstrates how adding momentum to SGD improves performance
and compares it to the Adam optimizer. We'll train three models:
1. SGD without momentum (baseline)
2. SGD with momentum
3. Adam (for comparison)
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
print("EXERCISE 3: SGD with Momentum vs. Adam")
print("=" * 70)

# Model 1: SGD without momentum (baseline)
print("\n--- Training Model 1: SGD (no momentum) ---")
model_sgd = create_model()
sgd_optimizer = keras.optimizers.SGD(learning_rate=0.01)
model_sgd.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history_sgd = model_sgd.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate
loss_sgd, acc_sgd = model_sgd.evaluate(X_test_scaled, y_test, verbose=0)

# Model 2: SGD with momentum
print("\n--- Training Model 2: SGD with Momentum (0.9) ---")
model_sgd_momentum = create_model()
sgd_momentum_optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model_sgd_momentum.compile(optimizer=sgd_momentum_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history_sgd_momentum = model_sgd_momentum.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate
loss_sgd_momentum, acc_sgd_momentum = model_sgd_momentum.evaluate(X_test_scaled, y_test, verbose=0)

# Model 3: Adam (for comparison)
print("\n--- Training Model 3: Adam ---")
model_adam = create_model()
model_adam.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_adam = model_adam.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate
loss_adam, acc_adam = model_adam.evaluate(X_test_scaled, y_test, verbose=0)

# Analysis
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

print("\nTest Accuracy:")
print(f"  1. SGD (no momentum):  {acc_sgd:.4f}")
print(f"  2. SGD with momentum:  {acc_sgd_momentum:.4f}")
print(f"  3. Adam:               {acc_adam:.4f}")

print("\nTest Loss:")
print(f"  1. SGD (no momentum):  {loss_sgd:.4f}")
print(f"  2. SGD with momentum:  {loss_sgd_momentum:.4f}")
print(f"  3. Adam:               {loss_adam:.4f}")

# Analyze convergence speed
final_train_acc_sgd = history_sgd.history['accuracy'][-1]
final_train_acc_sgd_momentum = history_sgd_momentum.history['accuracy'][-1]
final_train_acc_adam = history_adam.history['accuracy'][-1]

print("\nFinal Training Accuracy (convergence indicator):")
print(f"  1. SGD (no momentum):  {final_train_acc_sgd:.4f}")
print(f"  2. SGD with momentum:  {final_train_acc_sgd_momentum:.4f}")
print(f"  3. Adam:               {final_train_acc_adam:.4f}")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print("\n1. How does momentum improve SGD's performance?")
improvement = acc_sgd_momentum - acc_sgd
if improvement > 0.01:
    print(f"   → Momentum improved accuracy by {improvement:.4f}")
    print(f"   → Momentum helps SGD converge faster by 'remembering' which")
    print(f"      direction was working and continuing in that direction")
    print(f"   → This reduces oscillations and speeds up convergence")
elif improvement > 0:
    print(f"   → Momentum slightly improved accuracy by {improvement:.4f}")
else:
    print(f"   → Momentum didn't improve accuracy significantly on this dataset")
    print(f"      (This can happen with simple datasets or well-tuned learning rates)")

print("\n2. Does SGD with momentum match Adam's performance?")
if abs(acc_sgd_momentum - acc_adam) < 0.01:
    print(f"   → Yes! SGD with momentum matches Adam closely")
    print(f"      (Difference: {abs(acc_sgd_momentum - acc_adam):.4f})")
elif acc_sgd_momentum < acc_adam:
    print(f"   → No, Adam still outperforms SGD with momentum")
    print(f"      (Adam: {acc_adam:.4f} vs SGD+momentum: {acc_sgd_momentum:.4f})")
    print(f"   → Adam's adaptive learning rates give it an edge")
else:
    print(f"   → SGD with momentum actually outperformed Adam!")
    print(f"      (This can happen with careful tuning)")

print("\n3. Why might you still prefer Adam for most projects?")
print(f"   → Adam works well 'out of the box' with minimal tuning")
print(f"   → SGD requires careful selection of learning rate and momentum")
print(f"   → Adam adapts learning rates automatically for each parameter")
print(f"   → For production systems, Adam's reliability is valuable")
print(f"   → SGD is preferred when you need fine control or have")
print(f"      specific domain knowledge about your problem")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("Momentum:")
print("  • Helps SGD converge faster by reducing oscillations")
print("  • Remembers which direction was working and keeps going")
print("  • Typical value: 0.9 (90% of previous direction)")
print()
print("Adam vs SGD:")
print("  • Adam: Great default choice, works well without tuning")
print("  • SGD: Requires tuning but can be very effective when optimized")
print("  • SGD + momentum: Bridges the gap, still needs tuning")
print()
print("When to use what:")
print("  • Starting a new project? → Use Adam")
print("  • Need fine control? → Use SGD with momentum")
print("  • Production system? → Adam for reliability")
print("=" * 70)
