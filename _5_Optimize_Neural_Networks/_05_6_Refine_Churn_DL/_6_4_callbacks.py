"""
Keras Callbacks: Training Control and Optimization

Callbacks are powerful tools that let you "hook into" the training process.
They can automatically stop training, save checkpoints, adjust learning rates,
and more—all without manual intervention.

This script demonstrates three essential callbacks:
    1. EarlyStopping: Stops training when validation loss stops improving
       (prevents overfitting and saves time)
    2. ModelCheckpoint: Saves the best model weights during training
       (insurance against crashes and finding the peak performance)
    3. ReduceLROnPlateau: Reduces learning rate when learning stalls
       (helps escape plateaus and fine-tune towards minimum)

Key Takeaways:
    - Always use EarlyStopping for long training runs
    - ModelCheckpoint with save_best_only=True ensures you keep the best version
    - ReduceLROnPlateau helps optimization converge more precisely

Run with: conda run -n tf_env python _6_4_callbacks.py
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Re-using the dummy data preparation from the previous section
np.random.seed(42)
num_samples = 1000
features = np.random.rand(num_samples, 10) # 10 features
target = np.random.randint(0, 2, num_samples) # Binary target (churn/no churn)

gender = np.random.choice(['Male', 'Female'], num_samples)
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples)
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], num_samples)

df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(10)])
df['gender'] = gender
df['contract'] = contract
df['internet_service'] = internet_service
df['churn'] = target

df = pd.get_dummies(df, columns=['gender', 'contract', 'internet_service'], drop_first=True)

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a simple deep learning model for churn prediction
model = keras.Sequential([
    keras.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3), # Added dropout for regularization
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define Callbacks
# 1. Early Stopping: Stop if validation loss doesn't improve for 10 epochs
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss', # Metric to monitor
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True, # Restores model weights from the epoch with the best value of the monitored metric
    verbose=1
)

# 2. Model Checkpoint: Save the best model weights based on validation accuracy
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='best_churn_model.keras', # Path to save the model file
    monitor='val_accuracy',       # Metric to monitor
    save_best_only=True,          # Save only the best model found so far
    mode='max',                   # 'max' because we want to maximize accuracy
    verbose=1
)

# 3. Learning Rate Scheduler (ReduceLROnPlateau): Reduce LR if val_loss doesn't improve
lr_scheduler_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', # Metric to monitor
    factor=0.2,         # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=5,         # Number of epochs with no improvement after which learning rate will be reduced.
    min_lr=0.00001,     # Lower bound on the learning rate.
    verbose=1
)

# List of all callbacks to pass to the model.fit method
callbacks = [early_stopping_callback, model_checkpoint_callback, lr_scheduler_callback]

# Train the model with callbacks
history = model.fit(
    X_train_scaled, y_train,
    epochs=100, # Set a high number of epochs, EarlyStopping will stop it
    batch_size=32,
    validation_split=0.2, # Use part of the training data for validation
    callbacks=callbacks,
    verbose=0 # Set to 1 to see progress per epoch
)

print("\nTraining complete. Evaluating the best model...")
# The model will have the best weights restored by early_stopping_callback
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# You can also load the saved best model explicitly
# best_model_loaded = keras.models.load_model('best_churn_model.h5')
# loss_loaded, accuracy_loaded = best_model_loaded.evaluate(X_test_scaled, y_test, verbose=0)
# print(f"Loaded Best Model Test Loss: {loss_loaded:.4f}, Loaded Best Model Test Accuracy: {accuracy_loaded:.4f}")