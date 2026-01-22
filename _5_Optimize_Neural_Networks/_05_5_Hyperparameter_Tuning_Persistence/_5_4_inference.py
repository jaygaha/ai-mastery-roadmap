import keras
import numpy as np

# 1. Load the SavedModel directory as a Keras Layer
# 'serving_default' is the standard name for the prediction function
model_layer = keras.layers.TFSMLayer(
    'my_churn_model_savedmodel', 
    call_endpoint='serving_default'
)

print("SavedModel loaded successfully as a TFSMLayer.")

# 2. Example prediction
# TFSMLayer expects a batch of inputs
dummy_input = np.random.rand(1, 10)  # Ensure dim matches your training (e.g., 10)

# Then run prediction
prediction = model_layer(dummy_input)

# The output of TFSMLayer is typically a dictionary
# 'output_0' is the default key for the first output
print(f"Prediction for dummy input: {prediction['output_0']}")


