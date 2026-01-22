"""
Exercise 2: Random Search vs. Grid Search - Efficiency Challenge

Goal: Understand why "random" can sometimes be smarter (and faster) than "exhaustive".

Scenario: Training each model takes 10 minutes. 
- Calculate the total time for the Grid Search in Exercise 1.
- Calculate the total time for a Randomized Search with 20 iterations (`n_iter=20`).

Discussion: When would you choose Random over Grid?
"""

import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

# Assuming X_train has been defined with the correct input shape
input_dim = 10 # Example feature count for house price prediction

def create_regression_model(learning_rate=0.001, number_of_hidden_layers=1, neurons_per_layer=64):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,))) # Input layer

    for _ in range(number_of_hidden_layers):
        model.add(keras.layers.Dense(neurons_per_layer, activation='relu'))

    model.add(keras.layers.Dense(1)) # Output layer for regression

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

keras_regressor = KerasRegressor(
    model=create_regression_model,
    verbose=0,
    epochs=50 # Fixed epochs for each run during random search
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', keras_regressor)
])

"""
2. Define the param_distributions Dictionary:

We use scipy.stats.uniform for continuous distributions and scipy.stats.randint for discrete integer distributions. Remember the pipeline prefixes.
"""
param_distributions = {
    'regressor__model__learning_rate': uniform(loc=0.0001, scale=0.01 - 0.0001), # min=0.0001, max=0.01
    'regressor__batch_size': randint(16, 129), # min=16, max=128 (randint excludes the upper bound)
    'regressor__model__number_of_hidden_layers': randint(1, 6), # min=1, max=5
    'regressor__model__neurons_per_layer': randint(32, 257) # min=32, max=256
}

"""
Explanation of param_distributions:

    uniform(loc, scale): loc is the lower bound, scale is the range (upper bound - lower bound).
    randint(low, high): low is inclusive, high is exclusive. So randint(16, 129) samples integers from 16 up to (but not including) 129, effectively 16 to 128. The same logic applies to number_of_hidden_layers (1 to 5) and neurons_per_layer (32 to 256).
"""

"""
"""

if __name__ == "__main__":
    # Exercise 1 Recap
    grid_runs = 405 # From Exercise 1
    time_per_run = 10 # minutes
    total_grid_time = grid_runs * time_per_run

    # Exercise 2 Calculation
    n_iter = 20
    cv_folds = 5
    random_runs = n_iter * cv_folds
    total_random_time = random_runs * time_per_run

    print(f"--- Comparison ---")
    print(f"Grid Search Total Runs: {grid_runs}")
    print(f"Grid Search Total Time: {total_grid_time} minutes ({total_grid_time/60:.2f} hours)")
    print(f"\nRandomized Search Total Runs: {random_runs}")
    print(f"Randomized Search Total Time: {total_random_time} minutes ({total_random_time/60:.2f} hours)")
    
    print(f"\nConclusion:")
    print("RandomizedSearchCV is preferred when the hyperparameter space is large or when you have limited computational resources/time.")
    print(f"In this case, RandomizedSearchCV is {total_grid_time/total_random_time:.1f}x faster.")