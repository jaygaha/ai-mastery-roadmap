from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# Example of a simple Sequential model with L1 and L2 regularization

model = keras.Sequential([
    layers.Dense(
        128,
        activation='relu',
        # Apply L2 regularization to the kernel weights of this layer
        kernel_regularizer=regularizers.l2(0.001), # lambda = 0.001
        input_shape=(10,) # Assuming 10 input features
    ),
    layers.Dense(
        64,
        activation='relu',
        # Apply L1 regularization to the kernel weights of this layer
        kernel_regularizer=regularizers.l1(0.001) # lambda = 0.001
    ),
    layers.Dense(
        1,
        activation='sigmoid' # For binary classification like churn prediction
    )
])

# Summary to see the model architecture
model.summary()