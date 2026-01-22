"""
Exercise 1: Designing Your First Experiment

Goal: Imagine you're building a network to predict house prices. You need to decide which "knobs" (hyperparameters) to turn.

Task: Define a `param_grid` dictionary for `GridSearchCV` that explores:
    - learning_rate: 0.01, 0.001, 0.0001
    - batch_size: 32, 64, 128
    - number_of_hidden_layers: 1, 2, 3
    - neurons_per_layer: 32, 64, 128

Question: How many total model training runs would this grid search perform?
"""

import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasRegressor # Use KerasRegressor for regression tasks

# Assuming X_train has been defined with the correct input shape
# For demonstration, let's assume an input shape of 10 features for house price prediction
input_dim = 10 # Example feature count for house price prediction

def create_regression_model(learning_rate=0.001, number_of_hidden_layers=1, neurons_per_layer=64):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,))) # Input layer

    for i in range(number_of_hidden_layers):
        model.add(keras.layers.Dense(neurons_per_layer, activation='relu', name=f'hidden_layer_{i+1}'))

    # Output layer for regression (single neuron, no activation for linear output)
    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error') # Common loss for regression
    return model

# Wrap the Keras model using KerasRegressor for GridSearchCV
keras_regressor = KerasRegressor(
    model=create_regression_model,
    verbose=0,
    epochs=50 # Fixed epochs for each run during grid search
)

# Example for a pipeline if you were to use it (optional for this exercise, but good practice)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', keras_regressor)
])

"""
2. Define the param_grid Dictionary:
"""

param_grid = {
    'regressor__model__learning_rate': [0.01, 0.001, 0.0001],
    'regressor__batch_size': [32, 64, 128],
    'regressor__model__number_of_hidden_layers': [1, 2, 3],
    'regressor__model__neurons_per_layer': [32, 64, 128]
}

# If not using a pipeline, the keys would be simpler:
# param_grid_no_pipeline = {
#     'learning_rate': [0.01, 0.001, 0.0001],
#     'batch_size': [32, 64, 128],
#     'model__number_of_hidden_layers': [1, 2, 3], # model__ prefix for params passed to create_model
#     'model__neurons_per_layer': [32, 64, 128]
# }

"""

Explanation of param_grid keys:

    'regressor__model__learning_rate': regressor is the name of the KerasRegressor step in our pipeline. model__ indicates that learning_rate is an argument passed to the create_regression_model function (defined by the model parameter of KerasRegressor).
    'regressor__batch_size': batch_size is a direct parameter of the KerasRegressor wrapper itself.
    'regressor__model__number_of_hidden_layers': Similar to learning_rate, this is an argument for create_regression_model.
    'regressor__model__neurons_per_layer': Similar to learning_rate, this is an argument for create_regression_model.

"""

if __name__ == "__main__":
    
    learning_rate_opts = len(param_grid['regressor__model__learning_rate'])
    batch_size_opts = len(param_grid['regressor__batch_size'])
    hidden_layers_opts = len(param_grid['regressor__model__number_of_hidden_layers'])
    neurons_opts = len(param_grid['regressor__model__neurons_per_layer'])
    
    total_combinations = learning_rate_opts * batch_size_opts * hidden_layers_opts * neurons_opts
    cv_folds = 5
    total_runs = total_combinations * cv_folds

    print(f"Hyperparameter Options:")
    print(f"- Learning Rate: {learning_rate_opts}")
    print(f"- Batch Size: {batch_size_opts}")
    print(f"- Hidden Layers: {hidden_layers_opts}")
    print(f"- Neurons per Layer: {neurons_opts}")
    print(f"Total Unique Combinations: {total_combinations}")
    print(f"Total Training Runs (with CV={cv_folds}): {total_runs}")