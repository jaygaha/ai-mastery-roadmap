import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Generate synthetic data for regression
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, n_informative=5,
                               n_targets=1, noise=0.5, random_state=42)

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Build the MLP for regression
model_reg = keras.Sequential([
    keras.layers.Dense(units=128, activation='relu', input_shape=(X_train_reg_scaled.shape[1],)),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=1) # Single output neuron with linear activation (default) for regression
])

# Compile the model
model_reg.compile(optimizer='adam',
                  loss='mean_squared_error', # Use MSE for regression
                  metrics=['mae']) # Mean Absolute Error as a metric

# Train the model
print("\nTraining Regression Model:")
history_reg = model_reg.fit(X_train_reg_scaled, y_train_reg,
                            epochs=20, batch_size=32, validation_split=0.1, verbose=0)

# Evaluate the model
loss_reg, mae_reg = model_reg.evaluate(X_test_reg_scaled, y_test_reg, verbose=0)
print(f"Test Loss (Regression, MSE): {loss_reg:.4f}")
print(f"Test MAE (Regression): {mae_reg:.4f}")

# Make predictions
y_pred_reg = model_reg.predict(X_test_reg_scaled)

print("\nSample Regression Predictions:")
print(y_pred_reg[:5].flatten())
print("Sample Regression True Values:")
print(y_test_reg[:5].flatten())