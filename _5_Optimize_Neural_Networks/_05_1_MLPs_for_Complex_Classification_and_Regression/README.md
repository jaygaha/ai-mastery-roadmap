# Multi-Layer Perceptrons (MLPs) for Complex Classification and Regression

Multi-Layer Perceptrons (MLPs) are a type of neural network that adds more layers to the simple Perceptron we learned about earlier. By adding these extra "hidden" layers, MLPs can solve much harder problems, like recognizing handwriting or predicting stock prices. This is the foundation for most modern Deep Learning.

## Architecture of Multi-Layer Perceptrons

An MLP consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, in one layer connects to every node in the subsequent layer, forming a fully connected (dense) network. Data flows forward through the network, from the input layer, through the hidden layers, to the output layer.

### Input Layer

The input layer receives the raw features of the dataset. The number of neurons in this layer corresponds to the number of features in each input sample. These neurons do not perform any computations other than passing the input values to the next layer.

**Example:** Predicting house prices.
- **Features:** Square footage, number of bedrooms, zip code.
- **Input Layer:** 3 Neurons (one for each feature).

**Another Example:** Predicting air quality.
- **Features:** Temperature, humidity, pressure.
- **Input Layer:** 3 Neurons.

### Hidden Layers

Hidden layers are where the magic happens. They sit between the input and output layers.
- Each neuron here takes inputs, applies weights, adds a bias, and passes the result through an **activation function**.
- These layers allow the network to learn complex patterns (like shapes in an image or trends in data) instead of just simple straight lines.
- You can have one or many hidden layers. More layers = "Deep" Learning.

**Example 1 (Classification):** In a handwritten digit recognition task (0-9), the input layer receives pixel values of an image. The first hidden layer might learn to detect edges and simple shapes. A subsequent hidden layer could then combine these shapes to recognize parts of digits, eventually leading to the identification of the full digit in the output layer. 

**Example 2 (Regression):** Predicting a stock's future price using historical data. The input layer receives past stock prices, trading volumes, and market indicators. The first hidden layer might learn to identify short-term trends or volatility patterns. Deeper hidden layers could then combine these patterns to recognize more complex market behaviors, which are then used to predict the future price.

### Output Layer

The output layer produces the final prediction of the network. The number of neurons and the activation function used in this layer depend on the type of problem being solved.

- **For Classification:**

    - **Binary Classification (two classes):** Typically one output neuron with a Sigmoid activation function, producing a probability score between 0 and 1. If the probability is above a certain threshold (e.g., 0.5), it's classified as one class; otherwise, it's the other.
    - **Multi-class Classification (more than two classes):** Multiple output neurons, one for each class, with a Softmax activation function. Softmax converts the raw outputs (logits) into a probability distribution over the classes, where the sum of probabilities for all classes equals 1. The class with the highest probability is the model's prediction.

- **For Regression:**

    - Typically one output neuron with a linear (no) activation function, producing a continuous numerical value.

**Example (Customer Churn Prediction Case Study):** For our customer churn prediction, this is a binary classification problem. The output layer would have one neuron using a Sigmoid activation function to output the probability of a customer churning. A probability above, say, 0.5, would predict churn, while below 0.5 would predict no churn.

## Activation Functions in MLPs

Activation functions introduce non-linearity into the network, allowing MLPs to learn complex patterns that linear models cannot. Without activation functions, an MLP, no matter how many layers it has, would behave like a single linear model.

### ReLU (Rectified Linear Unit)

$f(x)=max(0,x)$

ReLU is a very popular activation function for hidden layers due to its computational efficiency and its ability to mitigate the vanishing gradient problem. It outputs the input directly if it's positive, otherwise it outputs zero.

### Sigmoid

$f(x)=\frac{1}{1+e^{-x}}$

The Sigmoid function squashes its input to a range between 0 and 1. It is primarily used in the output layer for binary classification problems, as it can be interpreted as a probability. For hidden layers, it has largely been replaced by ReLU due to the vanishing gradient issue for very large or very small inputs.

### Softmax

$f(x_i)=\frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}$

Softmax is used in the output layer for multi-class classification problems. It takes a vector of real numbers and normalizes them into a probability distribution, where each value is between 0 and 1, and the sum of all values equals 1. In the above formula $z_j$ is the input to the $j$-th output neuron, and $K$ is the number of classes.

**Real-world example:** Predicting customer sentiment (positive, negative, neutral) from text reviews. The output layer would have three neurons, one for each sentiment class, using Softmax to produce probabilities like [0.8, 0.1, 0.1] indicating 80% probability of positive sentiment.

## Implementing MLPs with Keras

Keras, a high-level API running on top of TensorFlow, simplifies the process of building and training neural networks. We will use the `Sequential` API for building our MLP.

### Building a Sequential Model

The `Sequential` model is a linear stack of layers.

```python
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Assume X_train, X_test, y_train, y_test are already prepared
# from the Customer Churn Prediction Case Study (Module 2, 3)

# For demonstration, let's create dummy data similar to the churn case study
# 10 features, 1000 samples for a binary classification problem
np.random.seed(42)
X = np.random.rand(1000, 10) * 100
y = (np.random.rand(1000) > 0.5).astype(int) # Binary target

# Simulate feature scaling as done in Module 2
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the MLP model
model = keras.Sequential([
    # Input layer implicitly defined by input_shape in the first hidden layer
    keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)), # First hidden layer
    keras.layers.Dense(units=32, activation='relu'), # Second hidden layer
    keras.layers.Dense(units=1, activation='sigmoid') # Output layer for binary classification
])

# Display model summary
model.summary()
```

- `keras.Sequential`: Initializes a linear stack of layers.
- `keras.layers.Dense`: Represents a fully connected layer.
    - `units`: The number of neurons in the layer.
    - `activation`: The activation function to use (relu, sigmoid, softmax).
    - `input_shape`: Required for the first layer to specify the shape of the input data (number of features). It's a tuple, e.g., (10,) for 10 features. For subsequent layers, Keras automatically infers the input shape.

### Compiling the Model

Before training, the model needs to be compiled. This involves specifying the optimizer, loss function, and metrics.

```python
# Compile the model
# For binary classification: 'binary_crossentropy' loss
# For regression: 'mean_squared_error' or 'mean_absolute_error'
# For multi-class classification: 'categorical_crossentropy' or 'sparse_categorical_crossentropy'

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) # For classification tasks

# If it were a regression problem:
# model_regression = keras.Sequential([
#     keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),
#     keras.layers.Dense(units=32, activation='relu'),
#     keras.layers.Dense(units=1) # Linear activation for regression output
# ])
# model_regression.compile(optimizer='adam',
#                          loss='mean_squared_error',
#                          metrics=['mae']) # Mean Absolute Error for regression
```


- `optimizer`: The algorithm used to update the model's weights during training (e.g., 'adam', 'sgd'). More details on optimizers will be covered in a later lesson.
- `loss`: The function that measures how well the model is performing, which the optimizer tries to minimize.
    - `'binary_crossentropy'` for binary classification.
    - `'categorical_crossentropy'` for multi-class classification where labels are one-hot encoded (e.g., [0, 1, 0] for class 1).
    - `'sparse_categorical_crossentropy'` for multi-class classification where labels are integers (e.g., 1 for class 1).
    - `'mean_squared_error'` (MSE) or `'mean_absolute_error'` (MAE) for regression.
- `metrics`: A list of metrics to be evaluated by the model during training and testing (e.g., 'accuracy' for classification, 'mae' for regression).

### Training the Model

Training an MLP involves feeding the model with input data and corresponding target outputs, allowing it to learn the mapping between them by adjusting its internal weights.

```python
# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# The 'history' object contains training loss and metrics for each epoch
print(history.history.keys())
```


- `X_train`, `y_train`: The training data and corresponding labels.
- `epochs`: The number of times the model will iterate over the entire training dataset. Each epoch represents one complete pass through the training data.
- `batch_size`: The number of samples per gradient update. The training data is divided into batches, and the model's weights are updated after processing each batch.
- `validation_split`: A fraction of the training data to be set aside as validation data. The model evaluates its performance on this validation set at the end of each epoch, which is useful for monitoring overfitting.

### Evaluating the Model

After training, the model's performance needs to be evaluated on unseen data to assess its generalization capabilities.

```python
# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
```

- `model.evaluate()`: Returns the loss value and metric values for the model in test mode.

### Making Predictions

Once trained and evaluated, the model can be used to make predictions on new, unseen data.

```python
# Make predictions on new data (e.g., X_test)
y_pred_proba = model.predict(X_test)

# For binary classification, convert probabilities to class labels
y_pred_classes = (y_pred_proba > 0.5).astype(int)

# Display some predictions
print("Sample Predictions (Probabilities):")
print(y_pred_proba[:5].flatten())
print("Sample Predicted Classes:")
print(y_pred_classes[:5].flatten())

# If it were a regression problem:
# y_pred_regression = model_regression.predict(X_test)
# print("Sample Regression Predictions:")
# print(y_pred_regression[:5].flatten())
```

- `model.predict()`: Generates predictions (probabilities for classification, raw values for regression) for the input samples.

## Practical Examples and Demonstrations

### Multi-Class Classification Example

Let's consider a scenario where we need to classify different types of customer feedback (e.g., "Feature Request," "Bug Report," "General Inquiry"). This is a multi-class classification problem.

[_5_2_1_example.py](./_5_2_1_example.py)

This example demonstrates how to set up an MLP for multi-class problems using `softmax` activation in the output layer and `categorical_crossentropy` as the loss function. The output probabilities for each class sum to 1, and `np.argmax` is used to get the predicted class.

### Regression Example

Consider predicting the satisfaction score (a continuous value from 0 to 10) of a customer based on their survey responses. This is a regression problem.

[_5_2_2_example.py](./_5_2_2_example.py)

This regression example uses a single output neuron with a default linear activation function, `mean_squared_error` for the loss, and `mean_absolute_error` as a metric. This setup is standard for predicting continuous numerical values.

## Exercises and Practice Activities

1. Modify Hidden Layers and Neurons (Churn Prediction): [_5_3_1_exercise.py](./_5_3_1_exercise.py) 
2. Experiment with Activation Functions: [_5_3_2_exercise.py](./_5_3_2_exercise.py) 
3. Implement Multi-Class Classification with Sparse Labels: [_5_3_3_exercise.py](./_5_3_3_exercise.py) 
4. Regression Model with Different Metrics: [_5_3_4_exercise.py](./_5_3_4_exercise.py) 

## Real-World Application

Multi-Layer Perceptrons are widely used across various industries due to their ability to model complex, non-linear relationships in tabular data.

### Financial Fraud Detection

Banks and financial institutions use MLPs to detect fraudulent transactions. The input features might include transaction amount, location, time of day, merchant category, customer's past spending habits, and number of recent transactions. The MLP learns intricate patterns that distinguish legitimate transactions from fraudulent ones. For instance, an MLP can identify suspicious activity where a customer's spending pattern suddenly deviates significantly, or a transaction occurs in an unusual location. This is typically a binary classification problem: fraudulent or not fraudulent.

### Medical Diagnosis and Prognosis

MLPs are applied in healthcare for tasks such as disease diagnosis. For example, based on a patient's medical history (age, gender, pre-existing conditions), lab test results (blood pressure, cholesterol levels, glucose levels), and genetic markers, an MLP can predict the likelihood of developing certain diseases like diabetes or heart disease. It can also be used for prognosis, estimating the severity or progression of a condition. This can be a multi-class classification (e.g., healthy, mild, severe) or binary classification (e.g., disease present or absent) problem. The MLP's ability to combine many input features allows for a more holistic assessment than traditional rule-based systems.

### Personalized Recommendation Systems

Although more advanced architectures like collaborative filtering are common, foundational MLPs can be part of recommendation systems. For example, in an e-commerce platform, an MLP can take a user's browsing history, purchase history, demographic information, and item attributes as input. It can then predict a numerical "liking" score (regression) for various unviewed products or classify (binary classification) whether the user will purchase a specific item. This allows for personalized product recommendations, enhancing user experience and driving sales.

## Next Steps and Future Learning Directions

This lesson has provided a deep dive into the architecture, implementation, and application of Multi-Layer Perceptrons for classification and regression tasks. You've gained hands-on experience building and training these foundational deep learning models using Keras. MLPs are a powerful step beyond single perceptrons, capable of learning complex non-linear mappings.

Building upon this, the next lessons in this module will focus on refining and optimizing these neural networks. You will explore critical techniques for effective training, such as understanding epochs, batch sizes, and the role of various optimizers like Adam and SGD. You will also learn how to combat common challenges in deep learning, particularly overfitting, by applying regularization strategies. These advanced concepts are essential for developing robust and high-performing neural network models.