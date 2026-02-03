"""
Exercise Solution: Using MobileNetV2 Architecture

This script demonstrates how to use MobileNetV2, an efficient, lightweight model designed for mobile devices.
It uses "Depthwise Separable Convolutions" to drastically reduce parameters.

Key features:
- Fast inference.
- Low memory footprint.
- Good accuracy-to-size ratio.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Set seed
tf.random.set_seed(42)

def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=10):
    print("Loading MobileNetV2...")
    # MobileNetV2 expects pixel values in [-1, 1], unlike VGG/ResNet which often use [0, 255] or centered.
    # However, the base model logic in tf.keras does some handling, but be aware of preprocessing.
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base
    for layer in base_model.layers:
        layer.trainable = False
        
    print(f"MobileNetV2 Base loaded with {len(base_model.layers)} layers.")
    
    # Add custom head
    x = base_model.output
    # GlobalAveragePooling is very standard for MobileNet
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

if __name__ == "__main__":
    # Define model
    model = build_mobilenet_model()
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Dummy Run
    print("Running dummy training step...")
    X_train = np.random.rand(10, 224, 224, 3).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, 10), 10)
    
    model.fit(X_train, y_train, epochs=1, verbose=1)
    print("MobileNetV2 Exercise Run Complete.")