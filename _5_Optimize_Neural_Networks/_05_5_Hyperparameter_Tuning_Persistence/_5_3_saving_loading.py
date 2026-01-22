# Step 1: Save the model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
# Step 2: Load the model
from tensorflow.keras.models import load_model

"""
STEP 1: Save the model
"""
input_dim = 10
# Create a dummy model (similar to churn prediction)
model = Sequential([
    Input(shape=(input_dim,)), 
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Dummy data for training
X_train_dummy = np.random.rand(100, 10)
y_train_dummy = np.random.randint(0, 2, 100)

# Train the model (e.g., for a few epochs for demonstration)
model.fit(X_train_dummy, y_train_dummy, epochs=5, batch_size=32, verbose=0)

# Save the entire model to an HDF5 file
model.save('my_churn_model.keras')
print("Model saved to my_churn_model.keras")

"""
STEP 2: Load the model
"""
# Load the entire model from the HDF5 file
loaded_model = load_model('my_churn_model.keras')

# Now you can use the loaded model for prediction or further training
print("Model loaded successfully.")
loaded_model.summary()

# Example prediction
dummy_input = np.random.rand(1, 10) # Single input for prediction
prediction = loaded_model.predict(dummy_input)
print(f"Prediction for dummy input: {prediction}")


"""
STEP 3: Save the model in the TensorFlow SavedModel format
"""
# Save the entire model in the TensorFlow SavedModel format
# Save as a SavedModel directory for deployment
model.export("my_churn_model_savedmodel")
print("Model saved to my_churn_model_savedmodel directory")

