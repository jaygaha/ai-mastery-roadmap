import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Generate dummy data for demonstration, similar to churn prediction features
np.random.seed(42)  # For reproducibility
num_samples = 10000
num_features = 10

# Create synthetic features (e.g., monthly charges, contract length, etc.)
# Scale for better feature distribution
X = np.random.rand(num_samples, num_features) * 100

# Create synthetic target variable (churn or not churn)
# Make it somewhat dependent on features to simulate a learnable pattern
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.rand(num_samples) * 10 > 70).astype(int)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (essential for neural networks to converge properly)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data
X_test_scaled = scaler.transform(X_test)        # Transform test data using training stats

# ============================================================================
# MODEL DEFINITION
# ============================================================================

def create_model():
    """
    Creates a simple feedforward neural network for binary classification.
    
    Architecture:
    - Input layer: 10 features
    - Hidden layer 1: 64 neurons with ReLU activation
    - Hidden layer 2: 32 neurons with ReLU activation
    - Output layer: 1 neuron with sigmoid activation (for binary classification)
    """
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    return model

print(f"Training data shape: {X_train_scaled.shape}")
print(f"Testing data shape: {X_test_scaled.shape}")


# ============================================================================
# EXPERIMENT 1: VARYING EPOCHS AND BATCH SIZE
# ============================================================================

# Model 1: Moderate epochs, standard batch size (balanced approach)
print("\n--- Training Model 1: Epochs=10, Batch Size=32 ---")
model_1 = create_model()
model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_1 = model_1.fit(X_train_scaled, y_train,
                        epochs=10, batch_size=32,
                        validation_split=0.1,  # Use 10% of training data for validation
                        verbose=1)


# Model 2: More epochs, smaller batch size (more frequent updates)
print("\n--- Training Model 2: Epochs=20, Batch Size=16 ---")
model_2 = create_model()
model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_2 = model_2.fit(X_train_scaled, y_train,
                        epochs=20, batch_size=16,
                        validation_split=0.1, verbose=1)


# Model 3: Fewer epochs, larger batch size (faster but fewer updates)
print("\n--- Training Model 3: Epochs=5, Batch Size=256 ---")
model_3 = create_model()
model_3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_3 = model_3.fit(X_train_scaled, y_train,
                        epochs=5, batch_size=256,
                        validation_split=0.1, verbose=1)


# ============================================================================
# EVALUATION: Compare different epoch and batch size configurations
# ============================================================================

print("\n--- Evaluation ---")
loss_1, acc_1 = model_1.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model 1 (Epochs=10, Batch=32) Test Accuracy: {acc_1:.4f}")

loss_2, acc_2 = model_2.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model 2 (Epochs=20, Batch=16) Test Accuracy: {acc_2:.4f}")

loss_3, acc_3 = model_3.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model 3 (Epochs=5, Batch=256) Test Accuracy: {acc_3:.4f}")


# ============================================================================
# EXPERIMENT 2: COMPARING OPTIMIZERS (Adam vs SGD)
# ============================================================================

# Model 4: Using Adam optimizer (adaptive learning rate)
print("\n--- Training Model 4: Optimizer='adam', Epochs=10, Batch Size=32 ---")
model_4 = create_model()
model_4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_4 = model_4.fit(X_train_scaled, y_train,
                        epochs=10, batch_size=32,
                        validation_split=0.1, verbose=1)


# Model 5: Using SGD optimizer with a fixed learning rate
print("\n--- Training Model 5: Optimizer='sgd', Epochs=10, Batch Size=32 ---")
model_5 = create_model()
# It's crucial to set a learning rate for SGD, as it's often more sensitive
sgd_optimizer = keras.optimizers.SGD(learning_rate=0.01)
model_5.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history_5 = model_5.fit(X_train_scaled, y_train,
                        epochs=10, batch_size=32,
                        validation_split=0.1, verbose=1)


# ============================================================================
# EVALUATION: Compare Adam vs SGD
# ============================================================================

print("\n--- Optimizer Evaluation ---")
loss_4, acc_4 = model_4.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model 4 (Adam) Test Accuracy: {acc_4:.4f}")

loss_5, acc_5 = model_5.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model 5 (SGD) Test Accuracy: {acc_5:.4f}")