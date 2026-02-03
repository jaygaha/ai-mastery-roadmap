"""
Exercise Solution: Experimenting with Freezing Layers

This script demonstrates the effect of freezing a different number of layers during fine-tuning.
In the main example, we froze the first 140 layers. Here, we try freezing only the first 50.

**Hypothesis**: Unfreezing more layers (freezing fewer) allows the model to adapt more 
deeply to the new dataset, but increases the risk of overfitting and training time.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Set seed
tf.random.set_seed(42)
np.random.seed(42)

# --- 1. Load & Prepare Base Model ---
print("Loading ResNet50...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Initial Freeze (Feature Extraction Phase)
for layer in base_model.layers:
    layer.trainable = False

# --- 2. Build Model ---
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# --- 3. Compile & Dummy Data ---
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("Generating dummy data...")
X_train = np.random.rand(100, 224, 224, 3).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, 2, 100), 2)
X_val = np.random.rand(20, 224, 224, 3).astype(np.float32)
y_val = tf.keras.utils.to_categorical(np.random.randint(0, 2, 20), 2)

# --- 4. Feature Extraction Training ---
print("\n--- Phase 1: Feature Extraction ---")
model.fit(X_train, y_train, epochs=2, verbose=1)

# --- 5. Fine-Tuning (The Exercise Part) ---
print("\n--- Phase 2: Fine-Tuning with Custom Freeze Depth ---")

# Unfreeze all
for layer in base_model.layers:
    layer.trainable = True

# EXERCISE MODIFICATION: Freeze only the first 50 layers (instead of 100 or 140)
fine_tune_at = 50
print(f"Freezing the first {fine_tune_at} layers (out of {len(base_model.layers)}).")
print("This means MORE layers are trainable than usual.")

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with low learning rate
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=2, validation_data=(X_val, y_val), verbose=1)

print("\nObservation: Notice the training speed and parameter count (if you ran summary).")