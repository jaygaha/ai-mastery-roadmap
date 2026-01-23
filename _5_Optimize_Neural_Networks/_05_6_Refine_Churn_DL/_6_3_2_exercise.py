"""
Exercise 2: Expand the Search Space

LEARNING OBJECTIVE:
Learn to create a more sophisticated hyperparameter search that includes
regularization techniques, normalization layers, and conditional model structures.

WHAT THIS EXERCISE ADDS:
    - Conditional L2 Regularization: Apply penalty to large weights (reduces overfitting)
    - Tunable Batch Normalization: Normalizes layer outputs with tunable momentum
    - Conditional Dropout: Random neuron deactivation (another overfitting preventer)
    - Dynamic Architecture: Optional pre-output dense layer

KEY INSIGHT:
By expanding the search space, Keras Tuner can discover more sophisticated
architectures that might significantly outperform simpler models. However,
larger search spaces need more trials to explore properly!

Run with: conda run -n tf_env python _6_3_2_exercise.py
"""

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load your churn data or use dummy data as before
try:
    df = pd.read_csv('../../_3_Core_Machine_Learning_Algorithms/_03_Logistic_Regression/customer_churn_preprocessed.csv')
except FileNotFoundError:
    print("churn_data.csv not found. Creating dummy data.")
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


def build_model_custom(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(X_train_scaled.shape[1],))) # Input layer

    # Tune the number of hidden layers
    for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=3, step=1)):
        # Conditionally add L2 regularization
        kernel_regularizer = None
        if hp.Boolean(f'use_l2_reg_{i}'):
            kernel_regularizer = keras.regularizers.l2(
                hp.Float(f'l2_lambda_{i}', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-4)
            )

        # Tune units in each hidden layer and activation
        model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                                     activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh']), # Limiting choices for faster tuning
                                     kernel_regularizer=kernel_regularizer))

        # Conditionally add Batch Normalization and tune its momentum
        if hp.Boolean(f'use_batch_norm_{i}'):
            model.add(keras.layers.BatchNormalization(
                momentum=hp.Float(f'bn_momentum_{i}', min_value=0.8, max_value=0.99, sampling='LINEAR', default=0.9)
            ))

        # Conditionally add Dropout
        if hp.Boolean(f'dropout_{i}'):
            model.add(keras.layers.Dropout(hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)))

    # Introduce a *potential* additional dense layer before the final output
    # This demonstrates tuning structural choices.
    if hp.Boolean('add_pre_output_dense_layer'):
        model.add(keras.layers.Dense(units=hp.Int('pre_output_dense_units', min_value=16, max_value=64, step=16),
                                     activation=hp.Choice('pre_output_activation', values=['relu', 'tanh'])))


    # Output layer for binary classification
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Tune learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

print("\n--- Running Hyperband with Custom Hyperparameters and Conditions ---")
tuner_custom = kt.Hyperband(build_model_custom,
                            objective='val_accuracy',
                            max_epochs=15, # Increased max_epochs slightly for more complex search
                            factor=3,
                            directory='my_dir',
                            project_name='churn_custom_hp')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) # Increased patience

tuner_custom.search(X_train_scaled, y_train,
                    epochs=75, # Total epochs for the entire tuning process
                    validation_split=0.2,
                    callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps_custom = tuner_custom.get_best_hyperparameters(num_trials=1)[0]

print("\n--- Best Hyperparameters Found ---")
print(f"Optimal Learning Rate: {best_hps_custom.get('learning_rate')}")
print(f"Optimal Number of Hidden Layers: {best_hps_custom.get('num_hidden_layers')}")

for i in range(best_hps_custom.get('num_hidden_layers')):
    print(f"\nLayer {i+1}:")
    print(f"  Units: {best_hps_custom.get(f'units_{i}')}")
    print(f"  Activation: {best_hps_custom.get(f'activation_{i}')}")
    if best_hps_custom.get(f'use_l2_reg_{i}'):
        print(f"  L2 Regularization Lambda: {best_hps_custom.get(f'l2_lambda_{i}'):.6f}")
    if best_hps_custom.get(f'use_batch_norm_{i}'):
        print(f"  Batch Normalization Momentum: {best_hps_custom.get(f'bn_momentum_{i}'):.2f}")
    if best_hps_custom.get(f'dropout_{i}'):
        print(f"  Dropout Rate: {best_hps_custom.get(f'dropout_rate_{i}')}")

if best_hps_custom.get('add_pre_output_dense_layer'):
    print("\nPre-Output Dense Layer Added:")
    print(f"  Units: {best_hps_custom.get('pre_output_dense_units')}")
    print(f"  Activation: {best_hps_custom.get('pre_output_activation')}")

# Retrieve the best model found
best_model_custom = tuner_custom.get_best_models(num_models=1)[0]

# Evaluate the best model on the test data
loss_custom, accuracy_custom = best_model_custom.evaluate(X_test_scaled, y_test)
print(f"\nCustom Tuner Test Loss: {loss_custom:.4f}, Test Accuracy: {accuracy_custom:.4f}")


"""
- This updated build_model_custom function now dynamically constructs the neural network based on various conditional hyperparameters. 
- Keras Tuner explores these possibilities, allowing it to discover more complex and potentially better architectures for churn prediction. 
- The sampling='LOG' for L2 regularization and sampling='LINEAR' for Batch Normalization momentum illustrate tuning continuous values over different
distributions.
"""