from tensorflow import keras
from tensorflow.keras import layers

# Example of a simple Sequential model with Dropout layers

model_with_dropout = keras.Sequential([
    layers.Dense(
        128,
        activation='relu',
        input_shape=(10,)
    ),
    # Apply dropout with a probability of 0.3 (30% of neurons will be dropped)
    layers.Dropout(0.3),
    layers.Dense(
        64,
        activation='relu'
    ),
    # Apply another dropout layer
    layers.Dropout(0.2),
    layers.Dense(
        1,
        activation='sigmoid'
    )
])

# Summary to see the model architecture
model_with_dropout.summary()