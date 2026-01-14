# Building and Training a Simple Feedforward Neural Network

In this lesson, we will move from theory to practice by building and training a simple Feedforward Neural Network (FNN) using **TensorFlow's Keras API**.

You have already learned about Perceptrons, activation functions, and how networks learn via backpropagation. Now, we will put it all together to build a model that can predict Customer Churn.

## 1. Setting Up the Environment

We will use **TensorFlow**, which includes **Keras** as its high-level API for building deep learning models easily.

Make sure, you have installed `tensorflow` and `pandas` in your environment.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(f"TensorFlow Version: {tf.__version__}")
```

## 2. Preparing the Data

Deep learning models require data to be numerical and often scaled. We will use the **Telco Customer Churn** dataset (from Module 2) instead of generating fake data.

### Loading and Preprocessing

```python
# Load the dataset
df = pd.read_csv('../../_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling/Telco-Customer-Churn.csv')

# Drop identifier column
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Handle missing numerical values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Encode target variable (Yes/No -> 1/0)
if df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Separate Features (X) and Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']
```

### Splitting and Scaling

Neural networks are sensitive to the scale of input data. Large values (like `MonthlyCharges` ~100) can dominate small values (like `Tenure` ~1). We use `StandardScaler` to normalize them.

```python
# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Input Shape: {X_train_scaled.shape}")
```

---

## 3. Building the Model

We will build a "Sequential" model, which acts like a stack of layers. Data flows from the first layer (input) to the last (output).

```python
model = keras.Sequential()

# 1. Input Layer + First Hidden Layer
# We specify 'input_shape' so the model knows how many features to expect.
model.add(layers.Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# 2. Second Hidden Layer
# We add another layer to help the model learn more complex patterns.
model.add(layers.Dense(units=32, activation='relu'))

# 3. Output Layer
# For binary classification (Yes/No), we use 1 neuron with 'sigmoid' activation.
# Sigmoid squashes the output between 0 and 1 (probability).
model.add(layers.Dense(units=1, activation='sigmoid'))

model.summary()
```

### Understanding `model.summary()`
- **Output Shape**: `(None, 64)` means the layer outputs 64 numbers for each sample. `None` is the batch size (variable).
- **Param #**: The number of weights and biases the model learns.

## 4. Compiling the Model

Before training, we must tell the model *how* to learn.

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

- **Optimizer (`adam`)**: The algorithm that updates the weights (like a smarter Gradient Descent).
- **Loss (`binary_crossentropy`)**: Measure of error for binary classification.
- **Metrics (`accuracy`)**: What we want to track during training.

## 5. Training

This is where the learning happens. We "fit" the model to the training data.

```python
history = model.fit(X_train_scaled, y_train,
                    epochs=50,          # Go through the entire dataset 50 times
                    batch_size=32,      # Update weights after every 32 samples
                    validation_split=0.1, # Use 10% of training data to check validation score
                    verbose=1)
```

- **Epoch**: One complete pass through the training data.
- **Batch Size**: How many samples processing before updating weights. Smaller batches often lead to better generalization but are slower.

## 6. Evaluation

Finally, we test the model on data it has **never seen** (the test set).

```python
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
```

### Conclusion
By adding layers and neurons, neural networks can approximate very complex functions. However, they also require careful tuning (network depth, learning rate, regularizations) which we will cover in the next module.

## Exercises

To practice, open the following files and solve the tasks:

- **[Exercise 1: Network Architectures](./_6_2_1_exercise.py)** - Try changing the number of layers and neurons.
- **[Exercise 2: Training Parameters](./_6_2_2_exercise.py)** - Experiment with `epochs` and `batch_size`.
