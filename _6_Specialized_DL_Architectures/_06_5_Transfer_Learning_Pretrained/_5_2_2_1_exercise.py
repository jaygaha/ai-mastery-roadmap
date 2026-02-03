"""
Exercise Solution: Using VGG16 Architecture

This script demonstrates how to successfully load the VGG16 model with pre-trained ImageNet weights.
VGG16 is a "classic" architecture known for its simplicity (just 3x3 convolutions) but high parameter count.

Key differences from ResNet:
- No residual connections.
- Much larger size (weights file is ~500MB).
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Set seed
tf.random.set_seed(42)

def build_vgg_model(input_shape=(224, 224, 3), num_classes=10):
    print("Loading VGG16...")
    # include_top=False removes the 1000-class ImageNet classifier
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base
    for layer in base_model.layers:
        layer.trainable = False
        
    print(f"VGG16 Base loaded with {len(base_model.layers)} layers.")
    
    # Add custom head
    # VGG16 output is 7x7x512. Flattening creates a large vector (25,088 elements).
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) # Common practice with VGG's dense layers
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

if __name__ == "__main__":
    # Define model
    model = build_vgg_model()
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Dummy Run
    print("Running dummy training step...")
    X_train = np.random.rand(10, 224, 224, 3).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, 10), 10)
    
    model.fit(X_train, y_train, epochs=1, verbose=1)
    print("VGG16 Exercise Run Complete.")