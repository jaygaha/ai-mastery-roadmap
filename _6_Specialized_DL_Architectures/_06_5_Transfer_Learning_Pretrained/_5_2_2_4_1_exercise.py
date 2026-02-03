"""
Exercise Solution: Adding Dropout for Regularization

This script demonstrates how to add Dropout layers to the classifier head.
Dropout is one of the most effective ways to prevent overfitting, especially when you have 
a small dataset relative to the model size (which is common in transfer learning).

How it works:
During training, it randomly "drops" (sets to zero) a percentage of neurons. 
This forces the network to learn redundant features and not rely on any single neuron.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Set seed
tf.random.set_seed(42)

def build_dropout_model(input_shape=(224, 224, 3), num_classes=10):
    print("Loading ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    # Use Flatten or GlobalPooling
    x = GlobalAveragePooling2D()(x)
    
    # 1st Dense Layer
    x = Dense(512, activation='relu')(x)
    
    # DROPOUT: 50% of neurons are dropped
    x = Dropout(0.5)(x) 
    
    # 2nd Dense Layer
    x = Dense(256, activation='relu')(x)
    
    # DROPOUT: 30% of neurons are dropped
    x = Dropout(0.3)(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

if __name__ == "__main__":
    model = build_dropout_model()
    
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("Running dummy training with Dropout...")
    X_train = np.random.rand(10, 224, 224, 3).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, 10), 10)
    
    model.fit(X_train, y_train, epochs=1, verbose=1)
    print("Dropout Exercise Run Complete.")