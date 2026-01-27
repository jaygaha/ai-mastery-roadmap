"""
Exercise 1: Experimenting with CNN Architecture Variations

This script demonstrates how different architectural choices in a Convolutional Neural Network (CNN) 
affect its performance (accuracy, loss) and training time.

We will compare three different models on the Fashion MNIST dataset:
1.  **Baseline CNN**: A standard, simple CNN architecture.
2.  **Deeper CNN**: A deeper network with Batch Normalization and Dropout.
3.  **Average Pooling CNN**: A network using Average Pooling instead of Max Pooling.

Key Concepts Demonstrated:
-   **Filters**: How many features the model learns at each layer.
-   **Kernel Size**: The size of the "window" used to detect features.
-   **Pooling**: Reducing the spatial dimensions (Max vs. Average).
-   **Regularization**: Using Dropout and Batch Normalization to improve generalization.
"""

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import numpy as np
import time
import matplotlib.pyplot as plt

# --- 1. Load and Preprocess Data (Fashion MNIST) ---
print("Loading and preprocessing data...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape data for CNN (add channel dimension: 28x28 -> 28x28x1)
# CNNs expect 4D input: (batch_size, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert labels to one-hot encoding
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# --- 2. Define Helper Function for Training ---
def build_and_train_model(name, model_fn, epochs=5, batch_size=64):
    """
    Builds, compiles, trains, and evaluates a model.
    """
    print(f"\n--- Training {name} ---")
    tf.keras.backend.clear_session() # Clear training state to ensure a fresh start

    model = model_fn(input_shape=x_train.shape[1:], num_classes=num_classes)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    start_time = time.time()
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1, # Use 10% of training data for validation
                        verbose=1)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"Training time for {name}: {training_time:.2f} seconds")
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy for {name}: {accuracy:.4f}")
    
    return {'model': model, 'history': history, 'accuracy': accuracy, 'time': training_time}

# --- 3. Define Architecture Variations ---

def build_baseline_cnn(input_shape, num_classes):
    """
    A standard CNN with 2 Convolutional blocks.
    Structure: Conv -> MaxPool -> Conv -> MaxPool -> Flatten -> Dense -> Output
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Classifier
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_deeper_cnn(input_shape, num_classes):
    """
    A deeper CNN that introduces Batch Normalization and Dropout.
    Batch Normalization helps stabilize training.
    Dropout helps prevent overfitting.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Block 3 (Deeper)
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        # No max pooling here to keep some spatial dimensions
        
        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3), # randomly drop 30% of neurons
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_avg_pooling_cnn(input_shape, num_classes):
    """
    Similar to baseline but uses AveragePooling2D.
    Average pooling smooths out the image features, which can be useful 
    but might lose sharp edge information compared to Max Pooling.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        AveragePooling2D((2, 2)), # Using Average Pooling
        
        Conv2D(64, (3, 3), activation='relu'),
        AveragePooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# --- 4. Run Experiments ---
# We use fewer epochs here for demonstration speed. 
# Increase epochs (e.g., to 10 or 20) for better convergence.
EPOCHS = 5 

results = {}

results['Baseline'] = build_and_train_model('Baseline CNN', build_baseline_cnn, epochs=EPOCHS)
results['Deeper'] = build_and_train_model('Deeper CNN', build_deeper_cnn, epochs=EPOCHS)
results['AvgPolling'] = build_and_train_model('Average Pooling CNN', build_avg_pooling_cnn, epochs=EPOCHS)

# --- 5. Compare and Visualize Results ---
print("\n--- Comparative Results ---")
for name, res in results.items():
    print(f"{name}: Accuracy={res['accuracy']:.4f}, Time={res['time']:.2f}s")

# Plot Validation Accuracy
plt.figure(figsize=(10, 5))
for name, res in results.items():
    plt.plot(res['history'].history['val_accuracy'], label=name)

plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()