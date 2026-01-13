# Setting Up TensorFlow and Keras

In this lesson, we'll get your machine ready for deep learning. We will install `TensorFlow` and `Keras`, which are the standard tools we'll be using to build and train neural networks.

## Installing TensorFlow and Keras

`TensorFlow` is the engine that does the heavy lifting, while `Keras` is the steering wheel—a user-friendly interface that makes building models much easier. In modern versions of TensorFlow, Keras is built right in.

### 1. Set Up a Virtual Environment (Recommended)

It's best practice to keep your deep learning projects separate from your other Python projects. This prevents version conflicts.

**Using default Python `venv`:**
```bash
python -m venv tf_env          # Create environment named 'tf_env'
source tf_env/bin/activate     # Activate on Mac/Linux
# tf_env\Scripts\activate      # Activate on Windows
```

**Using Anaconda:**
```bash
conda create -n tf_env python=3.10 # Create environment (Python 3.9-3.11 are reliable choices)
conda activate tf_env              # Activate environment
```

### 2. Install TensorFlow

Once your environment is active, install TensorFlow.

**For Standard Systems (Windows/Linux/Intel Mac)**
```bash
pip install tensorflow
```

**For Mac with Apple Silicon (M1/M2/M3/etc.)**
Apple provides a specialized plugin to make TensorFlow run fast on Mac GPUs.
```bash
pip install tensorflow
pip install tensorflow-metal
```

### 3. Verify Installation

Run this quick check in Python to make sure everything is working:

```python
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")
```

If you see a version number printed out, you're good to go!

---

## How Keras and TensorFlow Work Together

To put it simply: **TensorFlow** handles the math, and **Keras** handles the code structure.

-   **Deep Learning (Low Level):** Without Keras, you'd have to manually multiply matrices and manage complex math operations. It's powerful but hard to read and write.
-   **Deep Learning (High Level):** With Keras (`tf.keras`), you just say "I want a layer with 64 neurons," and it handles the math for you.

Keras is designed to be:
1.  **Simple:** Easy to learn and use.
2.  **Modular:** You snap "layers" together like building blocks.
3.  **Powerful:** It runs on top of TensorFlow, so it's fast and production-ready.

## Your First Neural Network

Let's look at a "Hello World" example. We will build a tiny brain that learns a simple math problem: `y = 2x - 1`.

### The Code
```python
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1. Define the Model
# We create a simple stack of layers ('Sequential').
# We add one layer ('Dense') with one neuron ('units=1').
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1])
])

# 2. Compile the Model
# We tell the model how to learn.
# Optimizer: 'adam' (a standard algorithm to improve the model)
# Loss: 'mean_squared_error' (how we measure mistake)
model.compile(optimizer='adam', loss='mean_squared_error')

# 3. Prepare Data
# The pattern is y = 2x - 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 4. Train the Model
# 'Fit' the model to the data for 500 loops (epochs)
model.fit(xs, ys, epochs=500)

# 5. Predict
# Ask the model: What is y when x is 10?
# The answer should be close to 19 (2*10 - 1)
print(model.predict(np.array([10.0])))
```

## Exercises

Get hands-on with these scripts to see how changes affect the model:

1.  **[Input Shape Experiment](./_4_2_1_exercise.py):** See what happens when the input data doesn't match what the model expects.
2.  **[Activation & Units](./_4_2_2_exercise.py):** Try changing the number of neurons and adding "activation functions" (like ReLU) to see how they change predictions.
3.  **[Optimizers & Loss](./_4_2_3_exercise.py):** Swap out the learning algorithm and error measurement to see if the model learns faster or slower.

---
**Next Up:** Now that your tools are ready, we will dive into the **Perceptron**—the fundamental building block of all neural networks.