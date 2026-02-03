"""
Exercise Solution: L1/L2 Regularization

This script demonstrates how to add L1/L2 weight regularization to the Dense layers.
Regularization adds a penalty to the loss function based on the size of the weights,
encouraging them to stay small (L2) or sparse (L1).

Difference:
- L2 (Ridge): Penalizes large weights. Good for general overfitting reduction.
- L1 (Lasso): Can drive weights to zero. Good for feature selection.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import numpy as np

# Set seed
tf.random.set_seed(42)

def build_regularized_model(input_shape=(224, 224, 3), num_classes=10):
    print("Loading ResNet50...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Layer with L2 Regularization
    x = Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001) # Penalty strength 0.001
    )(x)
    
    # Layer with L1 Regularization
    x = Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l1(0.0005) # Penalty strength 0.0005
    )(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

if __name__ == "__main__":
    model = build_regularized_model()
    
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("Running dummy training with L1/L2 Regularization...")
    X_train = np.random.rand(10, 224, 224, 3).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, 10), 10)
    
    model.fit(X_train, y_train, epochs=1, verbose=1)
    print("Regularization Exercise Run Complete.")