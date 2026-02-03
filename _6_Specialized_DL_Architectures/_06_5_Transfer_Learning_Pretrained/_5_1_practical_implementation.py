"""
Transfer Learning Practical Implementation with Keras

This script demonstrates the two main phases of Transfer Learning:
1.  **Feature Extraction**: Freezing the pre-trained base and training only the top classifier.
2.  **Fine-Tuning**: Unfreezing some layers of the base model to refine the features (with a low learning rate).

We use ResNet50 pre-trained on ImageNet as our base model.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# --- 1. Load a pre-trained model (ResNet50) ---
# We load ResNet50 pre-trained on ImageNet.
# `include_top=False` excludes the final classification layer (which has 1000 classes).
# We want to add our own classifier for our specific task (e.g., 2 classes).
print("Loading ResNet50...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Optional: Display the base model's summary to see its layers
# base_model.summary()

# --- 2. Feature Extraction Strategy: Freeze the base model layers ---
# We freeze the base model so its weights are NOT updated during the initial training.
# This prevents the large gradients from our randomly initialized classifier 
# from destroying the pre-trained weights.
for layer in base_model.layers:
    layer.trainable = False

# --- 3. Add custom classification layers on top of the base model ---
x = base_model.output
# GlobalAveragePooling2D is preferred over Flatten for modern CNNs because it:
# 1. Reduces overfitting by minimizing parameters.
# 2. Is robust to spatial translations.
x = GlobalAveragePooling2D()(x) 
x = Dense(1024, activation='relu')(x) # A large dense layer to learn complex combinations of features
predictions = Dense(2, activation='softmax')(x) # Final output for 2 classes

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Compile the model ---
# We use Adam with a standard learning rate for training the new top layers.
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary to verify that only the top layers are trainable
# model.summary()

# --- 5. Prepare data (Simulated) ---
# In a real project, you would use ImageDataGenerator or tf.data to load images from disk.
print("Generating dummy data...")
num_train_samples = 1000
num_val_samples = 200
batch_size = 32
num_classes = 2

# Dummy data: Random noise images
X_dummy_train = np.random.rand(num_train_samples, 224, 224, 3).astype(np.float32)
y_dummy_train = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_train_samples), num_classes)
X_dummy_val = np.random.rand(num_val_samples, 224, 224, 3).astype(np.float32)
y_dummy_val = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_val_samples), num_classes)

# --- 6. Train the new top layers (Feature Extraction) ---
print("\n--- PHASE 1: Feature Extraction (Training Top Layers) ---")
print("Base model is FROZEN. Only training the new Dense layers.")
model.fit(X_dummy_train, y_dummy_train,
          epochs=5,
          batch_size=batch_size,
          validation_data=(X_dummy_val, y_dummy_val))

# --- 7. Fine-tuning Strategy: Unfreeze and Refine ---
print("\n--- PHASE 2: Fine-Tuning ---")

# Step 1: Unfreeze the ENTIRE base model
for layer in base_model.layers:
    layer.trainable = True

# Step 2: Re-freeze the bottom N layers
# We typically want to keep the "low-level" features (edges, textures) fixed
# and only fine-tune the "high-level" features (shapes, objects).
# ResNet50 has 170+ layers. Let's freeze the first 140.
fine_tune_at = 140
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Number of layers in the base model: {len(base_model.layers)}")
print(f"Freezing the first {fine_tune_at} layers, fine-tuning the rest.")

# Step 3: Recompile with a VERY LOW learning rate
# This is CRITICAL. A high learning rate here will destroy the pre-trained weights.
# We want small, gentle updates.
model.compile(optimizer=Adam(learning_rate=0.00001), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 8. Continue training (Fine-tuning) ---
print("Continuing training...")
model.fit(X_dummy_train, y_dummy_train,
          epochs=5, 
          batch_size=batch_size,
          validation_data=(X_dummy_val, y_dummy_val))

# --- Evaluation ---
loss, accuracy = model.evaluate(X_dummy_val, y_dummy_val)
print(f"\nFinal Validation metrics result - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")