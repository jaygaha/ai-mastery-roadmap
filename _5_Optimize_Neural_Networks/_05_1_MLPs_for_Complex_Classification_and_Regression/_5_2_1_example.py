import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate synthetic data for 3-class classification
X_multi, y_multi = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                       n_redundant=0, n_classes=3, random_state=42)

# One-hot encode the target variable for 'categorical_crossentropy'
encoder = OneHotEncoder(sparse_output=False)
y_multi_encoded = encoder.fit_transform(y_multi.reshape(-1, 1))

# Split data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi_encoded, test_size=0.2, random_state=42
)

# Scale features (important for MLPs)
scaler_multi = StandardScaler()
X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
X_test_multi_scaled = scaler_multi.transform(X_test_multi)

# Build the MLP for multi-class classification
model_multi = keras.Sequential([
    keras.layers.Dense(units=128, activation='relu', input_shape=(X_train_multi_scaled.shape[1],)),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=3, activation='softmax') # 3 output neurons for 3 classes, with softmax
])

# Compile the model
model_multi.compile(optimizer='adam',
                    loss='categorical_crossentropy', # Use categorical_crossentropy for one-hot encoded labels
                    metrics=['accuracy'])

# Train the model
print("\nTraining Multi-Class Classification Model:")
history_multi = model_multi.fit(X_train_multi_scaled, y_train_multi,
                                epochs=20, batch_size=32, validation_split=0.1, verbose=0)

# Evaluate the model
loss_multi, accuracy_multi = model_multi.evaluate(X_test_multi_scaled, y_test_multi, verbose=0)
print(f"Test Loss (Multi-Class): {loss_multi:.4f}")
print(f"Test Accuracy (Multi-Class): {accuracy_multi:.4f}")

# Make predictions
y_pred_proba_multi = model_multi.predict(X_test_multi_scaled)
y_pred_classes_multi = np.argmax(y_pred_proba_multi, axis=1) # Get the index of the highest probability

print("\nSample Multi-Class Predictions (Probabilities):")
print(y_pred_proba_multi[:5])
print("Sample Multi-Class Predicted Labels (Original):")
print(encoder.inverse_transform(y_pred_proba_multi[:5]).flatten())
print("Sample Multi-Class True Labels (Original):")
print(encoder.inverse_transform(y_test_multi[:5]).flatten())