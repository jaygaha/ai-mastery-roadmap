import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
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


# Keras model wrapper for Scikit-learn's GridSearchCV
def create_model(learning_rate=0.001):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a Keras Classifier wrapper
keras_classifier = KerasClassifier(
    model=create_model, # Use 'model' argument instead of 'build_fn'
    verbose=0,
    # Pass arguments to the model's build function via 'model__param_name' in param_grid
)

# Define the parameter grid. Parameters for the 'model' function
# (like learning_rate) need to be prefixed with 'clf__model__'
# because 'clf' is the name of the KerasClassifier step in the pipeline.
# Parameters for the KerasClassifier wrapper itself (like batch_size, epochs)
# need to be prefixed with 'clf__'.
param_grid = {
    'clf__model__learning_rate': [0.001, 0.005, 0.01],
    'clf__batch_size': [32, 64, 128],
    'clf__epochs': [10, 20]
}

# Integrate with a pipeline for scaling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', keras_classifier) # Renamed to 'clf' for clarity
])

# Perform Grid Search
grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid, # Now correctly prefixed
                           scoring='accuracy',
                           cv=3,
                           verbose=1,
                           n_jobs=-1)

grid_search_result = grid_search.fit(X_train, y_train)

# Display best parameters and score
print(f"Best: {grid_search_result.best_score_:.4f} using {grid_search_result.best_params_}")

# Evaluate the best model on the test set
best_model = grid_search_result.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy of best model: {test_accuracy:.4f}")

# Access all results
# results_df = pd.DataFrame(grid_search_result.cv_results_)
# print(results_df[['param_model__learning_rate', 'param_model__batch_size', 'param_model__epochs', 'mean_test_score']])