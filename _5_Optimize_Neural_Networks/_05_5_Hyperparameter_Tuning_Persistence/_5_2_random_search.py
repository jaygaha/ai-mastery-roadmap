from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
import pandas as pd
import numpy as np

# Assume X_train, y_train are loaded and split from Module 2
# For demonstration, let's create dummy data if not already loaded
try:
    # This assumes you have X_train, y_train available from previous steps
    _ = X_train.head()
    _ = y_train.head()
except NameError:
    print("X_train, y_train not found. Generating dummy data for demonstration.")
    # Dummy data creation for example if not run through previous modules
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dummy data generated.")


# Keras model creation function.
# It now accepts `num_neurons` to be tuned.
def create_model(learning_rate=0.001, num_neurons=64): # Add num_neurons as an argument
    model = keras.Sequential([
        keras.layers.Dense(num_neurons, activation='relu', input_shape=(X_train.shape[1],)), # Use num_neurons
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_neurons // 2, activation='relu'), # Use num_neurons // 2 for second layer
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a Keras Classifier wrapper
keras_classifier = KerasClassifier(
    model=create_model,
    verbose=0,
    epochs=10 # Add a default or fixed number of epochs here for Random Search
)

# Define the parameter distributions to sample from
# Ensure all params are correctly prefixed for the pipeline steps.
# 'clf__' for parameters of KerasClassifier (like batch_size, epochs)
# 'clf__model__' for parameters passed to the create_model function (like learning_rate, num_neurons)
param_dist = {
    'clf__model__learning_rate': uniform(loc=0.0001, scale=0.01 - 0.0001), # Uniform distribution for learning rate
    'clf__batch_size': randint(16, 129), # Random integer for batch size
    'clf__model__num_neurons': randint(32, 129) # Random integer for number of neurons in the first layer
}

# Integrate with a pipeline for scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', keras_classifier) # Classifier step named 'clf'
])

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=pipeline, # Use the pipeline as the estimator
                                   param_distributions=param_dist,
                                   n_iter=10, # Number of parameter settings that are sampled
                                   cv=3,
                                   scoring='accuracy',
                                   verbose=1, # Increased verbosity for output during fit
                                   random_state=42,
                                   n_jobs=-1) # Use all available cores

# Fit the random search to the data
random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best accuracy: {random_search.best_score_}")

# Evaluate the best model on the test set
best_model = random_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy of best model: {test_accuracy:.4f}")