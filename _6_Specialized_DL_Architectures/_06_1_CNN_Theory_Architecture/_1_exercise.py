"""
Exercise: Convolutional Neural Networks (CNNs) - Hands-On Practice

This module provides practical exercises to reinforce CNN concepts from the theory lesson.
You'll calculate convolution outputs, implement max pooling, and build a simple CNN.

Prerequisites:
    - Understanding of CNN theory (convolution, pooling, filters)
    - NumPy basics
    - TensorFlow/Keras basics (from Module 4)

Usage:
    conda activate tf_env
    python _1_1_exercise.py
"""

import numpy as np

# ==============================================================================
# EXERCISE 1: Manual Convolution Calculation
# ==============================================================================


def calculate_output_dimensions(input_size: int, filter_size: int, stride: int, padding: str) -> int:
    """
    Calculate the output dimension after a convolution operation.

    The formula differs based on padding type:
    - "valid" (no padding): output = (input - filter) / stride + 1
    - "same" (with padding): output = input / stride (rounded up)

    Args:
        input_size: The height/width of the input (assuming square input)
        filter_size: The height/width of the filter (assuming square filter)
        stride: How many pixels the filter moves each step
        padding: Either "valid" (no padding) or "same" (zero padding)

    Returns:
        The output dimension (height == width for square outputs)

    Examples:
        >>> calculate_output_dimensions(10, 3, 1, "valid")
        8
        >>> calculate_output_dimensions(10, 3, 1, "same")
        10
    """
    if padding == "valid":
        # No padding: output shrinks
        output = (input_size - filter_size) // stride + 1
    elif padding == "same":
        # With padding: output stays same size (with stride=1)
        # For other strides, we round up
        output = (input_size + stride - 1) // stride
    else:
        raise ValueError(f"Unknown padding type: {padding}. Use 'valid' or 'same'.")

    return output


def exercise_1():
    """
    Exercise 1: Calculate Convolution Output Dimensions

    Given a 10x10 input with a 3x3 filter, calculate outputs for:
    a) stride=1, valid padding
    b) stride=1, same padding
    c) stride=2, valid padding
    """
    print("=" * 60)
    print("EXERCISE 1: Calculate Convolution Output Dimensions")
    print("=" * 60)

    input_size = 10
    filter_size = 3

    # a) Stride=1, Valid Padding
    output_a = calculate_output_dimensions(input_size, filter_size, stride=1, padding="valid")
    print(f"\na) Input: {input_size}x{input_size}, Filter: {filter_size}x{filter_size}")
    print(f"   Stride=1, Valid Padding")
    print(f"   Output: {output_a}x{output_a}")
    print(f"   Formula: ({input_size} - {filter_size}) / 1 + 1 = {output_a}")

    # b) Stride=1, Same Padding
    output_b = calculate_output_dimensions(input_size, filter_size, stride=1, padding="same")
    print(f"\nb) Input: {input_size}x{input_size}, Filter: {filter_size}x{filter_size}")
    print(f"   Stride=1, Same Padding")
    print(f"   Output: {output_b}x{output_b}")
    print(f"   With 'same' padding, output = input (for stride=1)")

    # c) Stride=2, Valid Padding
    output_c = calculate_output_dimensions(input_size, filter_size, stride=2, padding="valid")
    print(f"\nc) Input: {input_size}x{input_size}, Filter: {filter_size}x{filter_size}")
    print(f"   Stride=2, Valid Padding")
    print(f"   Output: {output_c}x{output_c}")
    print(f"   Formula: ({input_size} - {filter_size}) / 2 + 1 = {output_c}")


# ==============================================================================
# EXERCISE 2: Implement Max Pooling Manually
# ==============================================================================


def max_pool_2d(feature_map: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """
    Apply 2D max pooling to a feature map.

    This function demonstrates how max pooling works under the hood.
    For each non-overlapping region, we take the maximum value.

    Args:
        feature_map: 2D numpy array representing the input feature map
        pool_size: Size of the pooling window (assumes square)
        stride: How many pixels to move between pooling windows

    Returns:
        Pooled feature map (smaller than input)

    Example:
        >>> input_map = np.array([[1, 5, 2, 4],
        ...                       [3, 6, 7, 8],
        ...                       [9, 2, 1, 0],
        ...                       [4, 8, 3, 2]])
        >>> max_pool_2d(input_map, pool_size=2, stride=2)
        array([[6, 8],
               [9, 3]])
    """
    h, w = feature_map.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # Extract the pooling window
            row_start = i * stride
            row_end = row_start + pool_size
            col_start = j * stride
            col_end = col_start + pool_size

            window = feature_map[row_start:row_end, col_start:col_end]

            # Take the maximum value in the window
            output[i, j] = np.max(window)

    return output


def exercise_2():
    """
    Exercise 2: Max Pooling Operation

    Apply 2x2 max pooling with stride 2 to a 6x6 feature map.
    """
    print("\n" + "=" * 60)
    print("EXERCISE 2: Max Pooling Operation")
    print("=" * 60)

    # Define the 6x6 feature map from the README
    feature_map = np.array([
        [2, 4, 1, 5, 3, 6],
        [7, 1, 8, 2, 9, 0],
        [3, 5, 0, 4, 1, 7],
        [6, 2, 9, 1, 8, 3],
        [0, 8, 7, 3, 2, 5],
        [4, 1, 6, 9, 0, 2]
    ])

    print("\nInput Feature Map (6x6):")
    print(feature_map)

    # Apply max pooling
    pooled = max_pool_2d(feature_map, pool_size=2, stride=2)

    print(f"\nAfter 2x2 Max Pooling (stride=2):")
    print(f"Output size: {pooled.shape[0]}x{pooled.shape[1]}")
    print(pooled.astype(int))

    # Show step-by-step for understanding
    print("\nStep-by-step breakdown:")
    print("  Top-left [2,4,7,1] → max = 7")
    print("  Top-middle [1,5,8,2] → max = 8")
    print("  Top-right [3,6,9,0] → max = 9")
    print("  ... and so on")


# ==============================================================================
# EXERCISE 3: Build a Simple CNN with TensorFlow/Keras
# ==============================================================================


def exercise_3():
    """
    Exercise 3: Build and Inspect a CNN Architecture

    Build a simple CNN for image classification and inspect its structure.
    This demonstrates the typical Conv2D → ReLU → MaxPool → Dense pattern.
    """
    print("\n" + "=" * 60)
    print("EXERCISE 3: Build a Simple CNN")
    print("=" * 60)

    try:
        # Import TensorFlow (optional - only runs if TensorFlow is available)
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        print("\nBuilding a CNN for 28x28 grayscale images (like MNIST)...")

        # Build the model
        model = keras.Sequential([
            # Input layer
            keras.Input(shape=(28, 28, 1)),

            # First Convolutional Block
            # 32 filters of size 3x3, detects simple edges and patterns
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

            # Second Convolutional Block
            # 64 filters, learns more complex patterns from previous features
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

            # Flatten: Convert 2D feature maps to 1D
            layers.Flatten(name="flatten"),

            # Fully Connected: Combine features for classification
            layers.Dense(128, activation="relu", name="dense1"),
            layers.Dropout(0.5, name="dropout"),  # Prevent overfitting

            # Output: 10 classes (digits 0-9)
            layers.Dense(10, activation="softmax", name="output")
        ], name="SimpleCNN")

        # Print model summary
        print("\nModel Architecture:")
        model.summary()

        # Explain the architecture
        print("\n--- Architecture Explanation ---")
        print("Layer-by-layer breakdown:\n")

        print("1. Input (28×28×1): Grayscale images, 28 pixels × 28 pixels × 1 channel")
        print("2. Conv2D (32 filters, 3×3): Detects 32 different simple patterns")
        print("   → Output: 26×26×32 (shrinks due to valid padding)")
        print("3. MaxPool (2×2): Reduces to 13×13×32")
        print("4. Conv2D (64 filters, 3×3): Detects 64 complex patterns")
        print("   → Output: 11×11×64")
        print("5. MaxPool (2×2): Reduces to 5×5×64")
        print("6. Flatten: Converts to 1D vector of 5×5×64 = 1,600 values")
        print("7. Dense (128): Combines features")
        print("8. Dropout (0.5): Randomly drops 50% of neurons (training only)")
        print("9. Output (10): One probability per digit class")

    except ImportError:
        print("\n⚠️  TensorFlow not available. Skipping CNN building exercise.")
        print("   To run this exercise, activate the tf_env conda environment:")
        print("   conda activate tf_env")


# ==============================================================================
# EXERCISE 4: Calculate Trainable Parameters
# ==============================================================================


def calculate_conv_parameters(input_channels: int, num_filters: int, kernel_size: int) -> int:
    """
    Calculate the number of trainable parameters in a Conv2D layer.

    Each filter has:
    - Weights: kernel_height × kernel_width × input_channels
    - Bias: 1

    Args:
        input_channels: Number of channels in the input (e.g., 3 for RGB)
        num_filters: Number of filters in the layer
        kernel_size: Size of each filter (assumes square)

    Returns:
        Total number of trainable parameters
    """
    weights_per_filter = kernel_size * kernel_size * input_channels
    total_weights = weights_per_filter * num_filters
    total_biases = num_filters  # One bias per filter
    return total_weights + total_biases


def exercise_4():
    """
    Exercise 4: Calculate Trainable Parameters

    Calculate the total parameters for a Conv2D layer with:
    - Input: 28×28×128
    - Filter size: 3×3
    - Number of filters: 256
    """
    print("\n" + "=" * 60)
    print("EXERCISE 4: Calculate Trainable Parameters")
    print("=" * 60)

    input_channels = 128
    num_filters = 256
    kernel_size = 3

    total_params = calculate_conv_parameters(input_channels, num_filters, kernel_size)

    print(f"\nConv2D Layer Configuration:")
    print(f"  Input shape: 28×28×{input_channels}")
    print(f"  Filter size: {kernel_size}×{kernel_size}")
    print(f"  Number of filters: {num_filters}")

    print(f"\nCalculation:")
    weights_per_filter = kernel_size * kernel_size * input_channels
    print(f"  Weights per filter: {kernel_size}×{kernel_size}×{input_channels} = {weights_per_filter}")
    print(f"  Total weights: {weights_per_filter} × {num_filters} = {weights_per_filter * num_filters:,}")
    print(f"  Biases: 1 per filter × {num_filters} = {num_filters}")
    print(f"\n  Total parameters: {total_params:,}")


# ==============================================================================
# MAIN: Run All Exercises
# ==============================================================================


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   CNN EXERCISES - Hands-On Practice")
    print("=" * 60)
    print("\nThis script runs through 4 exercises covering CNN concepts:")
    print("  1. Calculate convolution output dimensions")
    print("  2. Implement max pooling manually")
    print("  3. Build a CNN with TensorFlow/Keras")
    print("  4. Calculate trainable parameters")

    # Run all exercises
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

    print("\n" + "=" * 60)
    print("   ALL EXERCISES COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Experiment with different input sizes and filter sizes")
    print("  - Try modifying the CNN architecture in Exercise 3")
    print("  - Move on to the next lesson to train a CNN on real data!")
    print()