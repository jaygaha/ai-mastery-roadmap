"""
Exercise: Visualizing CNN Filters

One key question in deep learning is: "What features is the model actually learning?"
In CNNs, the first few layers learn simple visual features like edges, corners, and textures.

This script creates a simple CNN model (similar to Exercise 1), trains it briefly, 
and then visualizes the filters (kernels) of the very first Conv2D layer.

Goal: Understand that convolutional filters are not "black boxes" but interpretable 
feature detectors.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Create a Simple Model ---
# We use a larger kernel size (5,5) in the first layer to make visualization clearer
model = Sequential([
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1), name='conv_layer_1'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])

# Ideally, you would train the model first so the filters learn meaningful patterns.
# For demonstration, we'll initialize them randomly (which still looks like noise)
# or load a pre-trained model if available.
# Let's quickly train it on a small subset of MNIST just to get *something* better than random noise.

print("Training a small model on MNIST to learn filters (this might take a few seconds)...")
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train[:1000].reshape(-1, 28, 28, 1).astype('float32') / 255.0 # Use only 1000 samples for speed
y_train = tf.keras.utils.to_categorical(y_train[:1000], 10)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, verbose=0) 
print("Training complete.")

# --- 2. Extract Filters ---
# Get the weights from the first convolutional layer
# The shape will be (kernel_height, kernel_width, input_channels, output_filters)
# e.g., (5, 5, 1, 32)
filters, biases = model.get_layer('conv_layer_1').get_weights()

# Normalize filter values to 0-1 range for plotting
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

num_filters = filters.shape[3]
print(f"Visualizing {num_filters} filters from the first layer...")

# --- 3. Plot Filters ---
plt.figure(figsize=(10, 10))
# Plot first 25 filters (5x5 grid)
n_filters_to_show = 25 

for i in range(n_filters_to_show):
    # The filter at index i is filters[:, :, 0, i] because input has 1 channel (grayscale)
    f = filters[:, :, 0, i]
    
    # Plot each filter
    ax = plt.subplot(5, 5, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(f, cmap='gray')
    plt.title(f"Filter {i+1}", fontsize=8)

plt.suptitle("First Logic Layer CNN Filters (Learned Features)")
plt.show()

print("\nExplanation:")
print("- Each small square is a 5x5 filter.")
print("- Light areas represent positive weights (excitation), dark areas represent negative weights (inhibition).")
print("- In a well-trained model, you might start to see edge detectors (lines at different angles) or blob detectors.")
