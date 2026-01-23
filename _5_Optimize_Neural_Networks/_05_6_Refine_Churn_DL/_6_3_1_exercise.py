"""
Exercise 1: Experiment with Tuner Algorithms

LEARNING OBJECTIVE:
Compare the three main hyperparameter tuning algorithms available in Keras Tuner
and understand their strengths and trade-offs.

TUNER COMPARISON:
┌─────────────────────┬─────────────────────────────────────────────────────────────┐
│ Algorithm           │ How It Works                                                │
├─────────────────────┼─────────────────────────────────────────────────────────────┤
│ RandomSearch        │ Randomly samples from the search space. Simple but          │
│                     │ surprisingly effective for high-dimensional spaces.         │
├─────────────────────┼─────────────────────────────────────────────────────────────┤
│ Hyperband           │ Uses "bandit-based" strategy to allocate resources.         │
│                     │ Trains many models briefly, keeps the best, trains longer.  │
│                     │ More efficient for deep learning where training is slow.    │
├─────────────────────┼─────────────────────────────────────────────────────────────┤
│ BayesianOptimization│ Builds a probabilistic model of the objective function.     │
│                     │ Intelligently picks next trials based on past results.      │
│                     │ Best when evaluation is expensive; can get stuck locally.   │
└─────────────────────┴─────────────────────────────────────────────────────────────┘

Run with: conda run -n tf_env python _6_3_1_exercise.py
"""

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('../../_3_Core_Machine_Learning_Algorithms/_03_Logistic_Regression/customer_churn_preprocessed.csv')
except FileNotFoundError:
    print("Data file not found. Creating dummy data.")
    np.random.seed(42)
    data_size = 1000
    df = pd.DataFrame({
        'Feature1': np.random.rand(data_size) * 100,
        'Feature2': np.random.randint(0, 5, data_size),
        'Feature3': np.random.normal(50, 10, data_size),
        'CategoricalFeature': np.random.choice(['A', 'B', 'C'], data_size),
        'Churn': np.random.randint(0, 2, data_size)
    })
    df = pd.get_dummies(df, columns=['CategoricalFeature'], drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(X_train_scaled.shape[1],)))
    for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=3, step=1)):
        model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                                     activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh', 'sigmoid'])))
        if hp.Boolean(f'dropout_{i}'):
            model.add(keras.layers.Dropout(hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

"""
RandomSearch

RandomSearch samples hyperparameters randomly from the defined search space. It's often a good baseline and can be surprisingly effective for 
high-dimensional spaces.

"""
print("\n--- Running RandomSearch ---")
tuner_rs = kt.RandomSearch(build_model,
                           objective='val_accuracy',
                           max_trials=10, # Number of different hyperparameter combinations to try
                           executions_per_trial=1, # How many times to train each model for robustness
                           directory='my_dir',
                           project_name='churn_rs')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner_rs.search(X_train_scaled, y_train,
                epochs=10, # Max epochs for each trial
                validation_split=0.2,
                callbacks=[stop_early])

best_hps_rs = tuner_rs.get_best_hyperparameters(num_trials=1)[0]
print(f"RandomSearch Best Learning Rate: {best_hps_rs.get('learning_rate')}")
best_model_rs = tuner_rs.get_best_models(num_models=1)[0]
loss_rs, accuracy_rs = best_model_rs.evaluate(X_test_scaled, y_test)
print(f"RandomSearch Test Accuracy: {accuracy_rs:.4f}")

"""

Hyperband

Hyperband is an advanced algorithm that uses a bandit-based strategy to efficiently allocate resources (like epochs) to promising configurations, 
pruning less promising ones early. It's often faster than RandomSearch for deep learning tasks.
"""

print("\n--- Running Hyperband ---")
tuner_hb = kt.Hyperband(build_model,
                        objective='val_accuracy',
                        max_epochs=10, # Max epochs for a single trial in a bracket
                        factor=3, # Reduction factor for Hyperband (e.g., train 1/3 of models for 3x epochs)
                        directory='my_dir',
                        project_name='churn_hb')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner_hb.search(X_train_scaled, y_train,
                epochs=50, # Total epochs for the entire tuning process
                validation_split=0.2,
                callbacks=[stop_early])

best_hps_hb = tuner_hb.get_best_hyperparameters(num_trials=1)[0]
print(f"Hyperband Best Learning Rate: {best_hps_hb.get('learning_rate')}")
best_model_hb = tuner_hb.get_best_models(num_models=1)[0]
loss_hb, accuracy_hb = best_model_hb.evaluate(X_test_scaled, y_test)
print(f"Hyperband Test Accuracy: {accuracy_hb:.4f}")

"""
BayesianOptimization

Bayesian Optimization constructs a probabilistic model of the objective function (validation accuracy) and uses it to select the most 
promising next hyperparameters to evaluate. This can be very efficient but can sometimes get stuck in local optima.
"""

print("\n--- Running BayesianOptimization ---")
tuner_bo = kt.BayesianOptimization(build_model,
                                 objective='val_accuracy',
                                 max_trials=10, # Number of different hyperparameter combinations to try
                                 num_initial_points=2, # Number of random points to sample before fitting the surrogate model
                                 executions_per_trial=1,
                                 directory='my_dir',
                                 project_name='churn_bo')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner_bo.search(X_train_scaled, y_train,
                epochs=10, # Max epochs for each trial
                validation_split=0.2,
                callbacks=[stop_early])

best_hps_bo = tuner_bo.get_best_hyperparameters(num_trials=1)[0]
print(f"BayesianOptimization Best Learning Rate: {best_hps_bo.get('learning_rate')}")
best_model_bo = tuner_bo.get_best_models(num_models=1)[0]
loss_bo, accuracy_bo = best_model_bo.evaluate(X_test_scaled, y_test)
print(f"BayesianOptimization Test Accuracy: {accuracy_bo:.4f}")

"""
By running these snippets sequentially, you can observe the different best hyperparameters found and the final test accuracies achieved by 
each search algorithm on your customer churn data. Note that execution times and results will vary based on your specific dataset, hardware, 
and the complexity of the search space. Remember to clear the my_dir directory or use different project names for each tuner type if you want to
 run them independently without interference from prior runs.
"""