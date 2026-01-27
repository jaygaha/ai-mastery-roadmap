"""
Exercise: Data Augmentation

Data augmentation is a technique used to artificially increase the diversity of your training set 
by applying random transformations (like rotation, zoom, and flips) to your existing images. 
This helps prevent overfitting and makes the model more robust.

This script demonstrates how to use Keras preprocessing layers to create an augmentation pipeline 
and visualizes the results on sample images from the CIFAR-10 dataset.
"""

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load Sample Data ---
print("Loading CIFAR-10 data...")
(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()

# Take a few sample images
sample_images = x_train[:8] 
sample_images = sample_images.astype('float32') / 255.0

# --- 2. Define Augmentation Pipeline ---
# These layers are only active during training (or when training=True is passed)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2), # Rotate by 20%
    layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    layers.RandomContrast(0.2),
])

# --- 3. Visualize Augmentation ---
print("Visualizing augmented images...")

plt.figure(figsize=(10, 10))

for i in range(8):
    # Original Image
    ax = plt.subplot(4, 4, 2*i + 1)
    plt.imshow(sample_images[i])
    plt.title("Original")
    plt.axis("off")
    
    # Augmented Image
    # We must pass training=True to activate the augmentation layers
    augmented_image = data_augmentation(tf.expand_dims(sample_images[i], 0), training=True)
    
    ax = plt.subplot(4, 4, 2*i + 2)
    plt.imshow(augmented_image[0])
    plt.title("Augmented")
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\nExplanation:")
print("- RandomFlip: Flips the image horizontally or vertically.")
print("- RandomRotation: Rotates the image by a random factor.")
print("- RandomZoom: Zooms in or out.")
print("- RandomContrast: Adjusts the contrast.")
print("\nIn a real model, you would include this 'data_augmentation' block as the first layer of your model.")
