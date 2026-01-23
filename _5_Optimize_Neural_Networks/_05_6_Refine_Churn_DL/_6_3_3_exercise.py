"""
Exercise 3: Optimize for F1-Score (Custom Metric Optimization)

LEARNING OBJECTIVE:
Learn how to optimize for metrics beyond standard accuracy and loss, specifically
the F1-score which is crucial for imbalanced classification problems like churn.

THE CHALLENGE:
Keras Tuner's built-in objective expects a metric available during training.
F1-score is typically calculated post-prediction, not during each training step.

OUR SOLUTION:
1. Create a custom F1-score metric that can be tracked during training
2. Use kt.Objective() to tell the tuner we want to MAXIMIZE this metric
3. Pass validation_data explicitly (not validation_split) to calculate on held-out data

WHY F1-SCORE MATTERS:
For churn prediction with imbalanced classes (few churners, many non-churners):
- Accuracy can be misleading (predicting "no churn" for everyone gives high accuracy)
- F1-score balances precision (avoiding false alarms) and recall (catching churners)

Run with: conda run -n tf_env python _6_3_3_exercise.py
"""

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
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

# Split into training + validation and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for column names if needed, though not strictly for Keras
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Define a custom F1 metric for Keras, primarily for monitoring
# This F1 metric will NOT be the direct optimization target for Keras Tuner
def f1_score_metric(y_true, y_pred):
    y_pred_bin = tf.round(y_pred) # Convert probabilities to binary predictions
    res = tf.py_function(lambda yt, yp: f1_score(yt, yp, average='weighted'),
                          inp=[y_true, y_pred_bin],
                          Tout=tf.float32)
    res.set_shape([]) # Ensure Keras knows the result is a scalar
    return res


def build_model_f1(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(X_train_scaled.shape[1],)))

    for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=3, step=1)):
        model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                                     activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh'])))
        if hp.Boolean(f'dropout_{i}'):
            model.add(keras.layers.Dropout(hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1_score_metric]) # Include custom F1 for monitoring

    return model

# To optimize for F1-score with Keras Tuner, we need to wrap the build_model_f1
# and the training process within a custom tuner class or directly calculate
# and return the F1-score after a brief training run within the `run_trial` method.
# For simplicity and directness, we will define a custom `HyperModel`
# which gives us control over the `fit` method and allows returning a custom objective.

class F1HyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.input_shape,)))

        for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=3, step=1)):
            model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                                         activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh'])))
            if hp.Boolean(f'dropout_{i}'):
                model.add(keras.layers.Dropout(hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)))

        model.add(keras.layers.Dense(1, activation='sigmoid'))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy']) # We don't need custom F1 here if we calculate it manually

        return model

    # Keras Tuner's default `fit` method uses `validation_data` directly for its objective.
    # To use F1-score, we must override `fit` in `HyperModel` to perform the prediction
    # and calculate F1. However, Hyperband and RandomSearch directly expect `objective`
    # to be a metric from the `model.fit` history.
    # A more common way for custom, non-Keras-native metrics is to define a custom `Tuner` class.
    # For this exercise, we'll try to use the `Hyperband` tuner but define the objective
    # as something like 'val_accuracy' and *then* manually pick the best F1 model from the trials.
    # Direct F1 optimization requires a custom tuner loop, which is more involved than `Hyperband`
    # or `RandomSearch` allow natively for non-Keras metrics.

# Reverting to standard Keras Tuner objective for practical reasons
# and will demonstrate how to *select* the best F1 model post-tuning.
print("\n--- Running Hyperband to find best model by F1-score (post-tuning selection) ---")

# We will monitor 'val_accuracy' during tuning as a proxy,
# then manually find the best model based on F1-score on the validation set.
tuner_f1 = kt.Hyperband(
    build_model_f1, # Use our original build_model_f1 that reports F1 as a metric
    objective=kt.Objective('val_f1_score_metric', direction='max'), # Explicitly state higher is better
    max_epochs=15,
    factor=3,
    directory='my_dir',
    project_name='churn_f1_opt'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Ensure to pass validation data separately if objective is on validation metrics
tuner_f1.search(X_train_scaled, y_train,
                epochs=75,
                validation_data=(X_val_scaled, y_val), # Use validation_data for objective
                callbacks=[stop_early])

# --- Post-tuning analysis for F1-score ---
# Keras Tuner stores results. We can load the best models and evaluate them.
# The `get_best_models` method will use the `objective` specified during tuner initialization.
# Since we set `objective='val_f1_score_metric'`, `get_best_models` will return the model
# that had the highest F1-score on the validation set during its training in the tuning process.

best_model_f1_optimized = tuner_f1.get_best_models(num_models=1)[0]
best_hps_f1_optimized = tuner_f1.get_best_hyperparameters(num_trials=1)[0]

print("\n--- Best Hyperparameters Found for F1-score Optimization ---")
print(f"Optimal Learning Rate: {best_hps_f1_optimized.get('learning_rate')}")
print(f"Optimal Number of Hidden Layers: {best_hps_f1_optimized.get('num_hidden_layers')}")

for i in range(best_hps_f1_optimized.get('num_hidden_layers')):
    print(f"\nLayer {i+1}:")
    print(f"  Units: {best_hps_f1_optimized.get(f'units_{i}')}")
    print(f"  Activation: {best_hps_f1_optimized.get(f'activation_{i}')}")
    if best_hps_f1_optimized.get(f'dropout_{i}'):
        print(f"  Dropout Rate: {best_hps_f1_optimized.get(f'dropout_rate_{i}')}")


# Evaluate the F1-optimized model on the test data
y_pred_proba = best_model_f1_optimized.predict(X_test_scaled)
y_pred_bin = (y_pred_proba > 0.5).astype(int) # Threshold at 0.5 for binary classification

test_f1 = f1_score(y_test, y_pred_bin, average='weighted')
loss_eval, accuracy_eval, _ = best_model_f1_optimized.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\nModel Optimized for F1-score:")
print(f"Test Loss: {loss_eval:.4f}, Test Accuracy: {accuracy_eval:.4f}, Test F1-Score (weighted): {test_f1:.4f}")
