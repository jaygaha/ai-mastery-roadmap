# Building an Image Classification Model with CNNs (TensorFlow Keras)

This lesson focuses on constructing an image classification model utilizing Convolutional Neural Networks (CNNs) with TensorFlow Keras. Building on the foundational understanding of neural networks from Module 4 and their optimization in Module 5, this lesson applies those principles to a specialized architecture designed for image data.

## Setting Up the Environment and Loading Data

Before building the CNN, the necessary libraries must be imported, and the dataset prepared. For image classification, datasets often consist of many image files organized into directories representing different classes. TensorFlow Keras includes utilities to load such datasets directly.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define parameters for data loading
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Load a dataset directly from directories
# For this example, we'll use a hypothetical dataset structure.
# Imagine you have a 'train' directory and a 'validation' directory,
# each containing subdirectories for each image class (e.g., 'cat', 'dog').

# Example using a small built-in dataset for demonstration if a custom one isn't available:
# Here we'll simulate loading a dataset like CIFAR-10 which is common for image classification.
# For a real project, you would replace this with tf.keras.utils.image_dataset_from_directory.
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# In a real-world scenario with image_dataset_from_directory,
# your data loading would look more like this:
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     'path/to/your/train_data',
#     labels='inferred',
#     label_mode='int', # or 'categorical'
#     image_size=(IMG_HEIGHT, IMG_WIDTH),
#     interpolation='nearest',
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )
# val_ds = tf.keras.utils.image_dataset_from_directory(
#     'path/to/your/validation_data',
#     labels='inferred',
#     label_mode='int',
#     image_size=(IMG_HEIGHT, IMG_WIDTH),
#     interpolation='nearest',
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )
# And then you would process val_ds, etc.

# For CIFAR-10, the images are 32x32x3. Let's adapt our target size for demonstration.
# If we were using tf.keras.utils.image_dataset_from_directory, resizing would happen automatically.
# For this example, we will resize them to our target IMG_HEIGHT and IMG_WIDTH.
# This step is often done as part of the data pipeline or within the model if using tf.keras.layers.Resizing
train_images_resized = tf.image.resize(train_images, (IMG_HEIGHT, IMG_WIDTH)).numpy()
test_images_resized = tf.image.resize(test_images, (IMG_HEIGHT, IMG_WIDTH)).numpy()

# Determine number of classes (CIFAR-10 has 10 classes)
num_classes = len(np.unique(train_labels))
print(f"Number of classes: {num_classes}")

# Display a sample image
plt.figure(figsize=(4, 4))
plt.imshow(train_images_resized[0])
plt.title(f"Sample Image (Label: {train_labels[0][0]})")
plt.axis('off')
plt.show()
```

The `tf.keras.utils.image_dataset_from_directory` utility is critical for efficiently loading image data, handling resizing, and creating batches. Normalization scales pixel values from 0-255 to 0-1, which aids neural network training by preventing large input values from causing issues with gradient descent.

## Building the CNN Model Architecture

A Convolutional Neural Network (CNN) is specifically designed to process pixel data in images. Its architecture typically consists of convolutional layers, pooling layers, and fully connected (dense) layers.

### Convolutional Layers (Conv2D)

Convolutional layers are the core building blocks of CNNs. They apply a set of learnable filters (or kernels) to the input image, producing feature maps. Each filter specializes in detecting different features, such as edges, textures, or specific patterns. **Think of these filters as a team of detectives:** one looks for horizontal lines, another for curves, and another for color transitions. As we go deeper into the network, these detectives combine their findings to recognize complex objects like "ears" or "wheels."

- **filters:** The number of output filters in the convolution. Each filter learns a different feature. More filters allow the model to learn a richer set of features.
- **kernel_size:** The dimensions of the convolutional window. Common sizes are (3, 3) or (5, 5). Smaller kernels are good for capturing fine-grained details, while larger kernels capture broader patterns.
- **activation:** The activation function applied to the output of the convolution. `relu` (Rectified Linear Unit) is commonly used for its computational efficiency and ability to mitigate the vanishing gradient problem.
- **input_shape:** Required for the first layer, specifying the dimensions of the input images (height, width, channels). For color images, channels will be 3 (RGB); for grayscale, it's 1.

### Pooling Layers (MaxPooling2D)

Pooling layers reduce the spatial dimensions (width and height) of the feature maps. Imagine you have a high-resolution photo and you shrink it down—you lose some detail, but the main subjects are still recognizable. This helps to:

- **Reduce computational cost:** Fewer parameters mean faster training.
- **Control overfitting:** By summarizing features, the model becomes more robust to small variations in the input.
- **Achieve translational invariance:** The model becomes less sensitive to the exact position of a feature in the input image (e.g., a cat in the corner is still a cat).

- **pool_size:** The size of the window to take the maximum (or average) over. A (2, 2) pool size reduces the feature map dimensions by half.

### Flatten Layer (Flatten)

After several convolutional and pooling layers, the high-dimensional feature maps (which are 3D volumes) need to be converted into a 1D vector to be fed into fully connected layers. This is like taking a stack of papers and laying them out in a single long line so a standard neural network can read them.

### Dense Layers (Dense)

These are standard feedforward neural network layers as introduced in Module 4. They take the flattened feature vector and perform the final classification.

- **units:** The number of neurons in the layer. The final Dense layer's units should match the number of output classes.
- **activation:** The activation function. For binary classification, sigmoid is used; for multi-class classification, softmax is used to output probabilities for each class.

```python
model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Third Convolutional Block (optional, for deeper models)
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten the 3D output to 1D to feed into Dense layers
    layers.Flatten(),
    
    # Fully Connected Layers
    layers.Dense(units=128, activation='relu'),
    layers.Dropout(0.5), # Dropout for regularization (from Module 5)
    layers.Dense(units=num_classes, activation='softmax') # Output layer for multi-class classification
])

# Display the model summary
model.summary()
```

The `Dropout` layer (introduced in Module 5) is added to prevent overfitting by randomly setting a fraction of input units to zero at each update during training, which helps ensure that no single neuron becomes too reliant on specific features.

## Compiling and Training the Model

Once the model architecture is defined, it needs to be compiled and trained.

### Compilation

The compilation step configures the model for training.

- **optimizer**: The algorithm used to update the weights during training. `Adam` is a popular choice due to its efficiency and good performance across a wide range of tasks (discussed in Module 5).
- **loss**: The loss function measures how well the model predicts the correct labels. For multi-class classification with integer labels, `sparse_categorical_crossentropy` is appropriate. If labels are one-hot encoded, `categorical_crossentropy` would be used.
- **metrics**: Metrics are used to monitor the training and testing steps. `accuracy` is a common metric for classification tasks.

### Training

The `fit` method trains the model on the training data.

- **x and y**: The training input features (images) and corresponding labels.
- **epochs**: The number of times the model will iterate over the entire training dataset.
- **batch_size**: The number of samples processed before the model's parameters are updated.
- **validation_data**: A tuple of (validation images, validation labels) or a tf.data dataset. The model evaluates its performance on this data at the end of each epoch, providing insight into its generalization ability and helping detect overfitting.

```python
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Use sparse_categorical_crossentropy for integer labels
              metrics=['accuracy'])

# Train the model
# Using resized CIFAR-10 data for demonstration
history = model.fit(train_images_resized, train_labels,
                    epochs=10, # Number of epochs can be adjusted
                    batch_size=BATCH_SIZE,
                    validation_data=(test_images_resized, test_labels)) # Use test set as validation for simplicity here
                                                                       # In a real project, you'd have a separate validation set.
```

The `history` object returned by `model.fit` contains training and validation loss and metric values for each epoch, which can be useful for visualizing training progress.

## Evaluating and Visualizing Model Performance

After training, it is crucial to evaluate the model's performance on unseen data and visualize the training process.

### Evaluation

The `evaluate` method computes the loss and metrics for the given data.

```python
# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images_resized, test_labels, verbose=2)
print(f"\nTest accuracy: {test_accuracy*100:.2f}%")
```

### Visualization

Plotting the training and validation accuracy and loss curves helps to understand if the model is learning effectively, overfitting, or underfitting.

- **Overfitting**: Training accuracy is much higher than validation accuracy, and validation loss starts increasing while training loss continues to decrease.
- **Underfitting**: Both training and validation accuracy are low, indicating the model has not learned enough from the data.

```python
# Plot training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```

## Making Predictions

After training and evaluating the model, it can be used to make predictions on new, unseen images.

```python
# Get predictions for a batch of test images
predictions = model.predict(test_images_resized[:5])
predicted_labels = np.argmax(predictions, axis=1)

# Display the first 5 test images and their predicted labels
class_names = [f'class_{i}' for i in range(num_classes)] # Placeholder names for CIFAR-10
# If using image_dataset_from_directory, you would get class_names from train_ds.class_names

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(test_images_resized[i])
    plt.title(f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[test_labels[i][0]]}")
    plt.axis('off')
plt.suptitle('Predictions on Test Images')
plt.show()
```

The ``np.argmax`` function is used to find the index of the highest probability in the output of the softmax activation, which corresponds to the predicted class.

## Exercises

1. **Experiment with Architecture Variations**: Modify the number of `filters` in the ``Conv2D`` layers (e.g., 16, 32, 64) and the ``kernel_size`` (e.g., (5, 5) or (7, 7)). Observe how these changes affect the model's performance and training time.
    - [Run _2_1_exercise.py](_2_1_exercise.py)

2. **Adjust Pooling Layers**: Change the `pool_size` of `MaxPooling2D` layers (e.g., (3, 3)). What happens to the output shape of the layers and the model's ability to learn features?
    > **SOLUTION:** changing `pool_size` to (3, 3) increases the reduction in spatial dimensions. This leads to smaller feature maps (requiring fewer parameters in subsequent layers) but might cause the loss of fine-grained details, potentially hurting accuracy on complex images.

3. **Explore Data Augmentation**: Use `tf.keras.layers.RandomFlip`, `RandomRotation`, and `RandomZoom` to create a data augmentation pipeline. Augmentation helps the model generalize better by artificially increasing the diversity of your training set.
    - [Run _2_2_data_augmentation.py](_2_2_data_augmentation.py)

4. **Visualize CNN Filters**: Understand what your network is actually "seeing." Use the provided script to visualize the filters of the first convolutional layer.
    - [Run _2_3_visualize_filters.py](_2_3_visualize_filters.py)

5. **Regularization Impact**: Increase or decrease the `dropout` rate in the `Dropout` layer (e.g., 0.2, 0.7). Retrain the model and analyze the impact on overfitting by comparing training and validation accuracy/loss plots.
    > **SOLUTION:** A lower dropout rate (e.g., 0.2) retains more information but might not prevent overfitting as effectively. A higher rate (e.g., 0.7) forces the network to be more robust but can lead to underfitting if the model doesn't have enough capacity to learn.

## Next Steps and Future Learning Directions

In this lesson, you constructed and trained a basic CNN for image classification. While this provides a strong foundation, the world of specialized deep learning architectures offers many advanced techniques. Upcoming lessons will explore Recurrent Neural Networks (RNNs) for sequence data, which are fundamentally different from CNNs and are crucial for tasks like time series forecasting or natural language processing. Following that, transfer learning will be introduced, a powerful technique that leverages pre-trained models to significantly speed up development and improve performance on new datasets, especially when limited training data is available. This often involves fine-tuning existing CNNs that have been trained on vast image datasets.