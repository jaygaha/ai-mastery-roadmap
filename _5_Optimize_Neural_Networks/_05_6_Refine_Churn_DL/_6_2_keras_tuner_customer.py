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

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features (assuming some features need scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame if you need feature names later, though not strictly necessary for Keras input
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(X_train_scaled.shape[1],))) # Input layer

    # Tune the number of hidden layers
    for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=3, step=1)):
        # Tune units in each hidden layer
        model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                                     # Tune activation function
                                     activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh', 'sigmoid'])))
        # Add dropout conditionally
        if hp.Boolean(f'dropout_{i}'):
            model.add(keras.layers.Dropout(hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)))

    # Output layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Tune learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the Keras Tuner
# We'll use Hyperband, which is efficient for deep learning
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10, # Max epochs for a single trial
                     factor=3, # Reduction factor for Hyperband
                     directory='my_dir', # Directory to store results
                     project_name='churn_prediction_tuning')

# Define a callback to stop training early if validation accuracy doesn't improve
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Search for the best hyperparameters
tuner.search(X_train_scaled, y_train,
             epochs=50, # Total epochs to run for the tuning process across all trials
             validation_split=0.2,
             callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of hidden layers is {best_hps.get('num_hidden_layers')}.
The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
And here are some optimal unit counts and activation functions per layer:
""")
for i in range(best_hps.get('num_hidden_layers')):
    print(f"Layer {i+1}: Units = {best_hps.get(f'units_{i}')}, Activation = {best_hps.get(f'activation_{i}')}")
    if best_hps.get(f'dropout_{i}'):
        print(f"         Dropout Rate = {best_hps.get(f'dropout_rate_{i}')}")

# Retrieve the best model found
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model on the test data
loss, accuracy = best_model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# You can then train this best_model on the full training dataset for more epochs
# if desired, or save it directly.