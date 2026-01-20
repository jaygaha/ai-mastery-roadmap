"""
Exercise

3. Customer Churn Case Study - Regularization Strategy
"""

import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Suppress TensorFlow warnings
tf.keras.backend.set_image_data_format('channels_last') 

# --- DATA LOADING OR GENERATION ---
# We try to load the real data. If it's missing, we create a "mock" dataset so you can still run the code.

def load_or_create_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the file from the previous module
    data_path = os.path.join(script_dir, '../../_3_Core_Machine_Learning_Algorithms/_03_Logistic_Regression/customer_churn_preprocessed.csv')
    
    if os.path.exists(data_path):
        print(f"Loading real data from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"WARNING: Data file not found at {data_path}")
        print("Generating dummy data so you can see the code in action...")
        np.random.seed(42)
        data_size = 1000
        # Create synthetic features that mimic customer data
        df = pd.DataFrame({
            'CreditScore': np.random.normal(650, 100, data_size),
            'Age': np.random.randint(18, 90, data_size),
            'Tenure': np.random.randint(0, 10, data_size),
            'Balance': np.random.normal(50000, 20000, data_size),
            'NumOfProducts': np.random.randint(1, 4, data_size),
            'HasCrCard': np.random.randint(0, 2, data_size),
            'IsActiveMember': np.random.randint(0, 2, data_size),
            'EstimatedSalary': np.random.uniform(20000, 150000, data_size),
            'Churn': np.random.randint(0, 2, data_size) # Target
        })
        # Add some "noise" features to make regularization more useful
        df['NoiseFeature1'] = np.random.rand(data_size)
        df['NoiseFeature2'] = np.random.rand(data_size)
        print("Dummy data created successfully.")
    return df

# 1. Prepare Data
df = load_or_create_data()

# Separate Features (X) and Target (y)
if 'Churn' in df.columns:
    X = df.drop('Churn', axis=1)
    y = df['Churn']
else:
    # Fallback if the CSV has different column names (adjust as needed for your specific CSV)
    print("Column 'Churn' not found. assuming last column is target.")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

# Split data: 70% Train, 15% Validation, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features (Critical for Neural Networks!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

input_dim = X_train_scaled.shape[1]

# --- HELPER FUNCTIONS ---

def plot_history(histories, title):
    plt.figure(figsize=(15, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} Train')
        plt.plot(history.history['val_accuracy'], label=f'{name} Val', linestyle='--')
    plt.title(f'{title} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Loss
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} Train')
        plt.plot(history.history['val_loss'], label=f'{name} Val', linestyle='--')
    plt.title(f'{title} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, model_name):
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}") # F1 is often more important for Churn (imbalanced classes)
    return {'accuracy': acc, 'f1': f1}

# --- TRAINING CONFIGURATION ---
epochs = 50
batch_size = 64
verbose = 0 
all_histories = {}
results = {}

# 2. Experiment: Compare 3 Strategies

# --- Strategy A: Base Model (No Regularization) ---
print("\nTraining Base Model...")
model_base = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_base.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
all_histories['Base'] = model_base.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_scaled, y_val), verbose=verbose)
results['Base'] = evaluate_model(model_base, X_test_scaled, y_test, "Base Model")

# --- Strategy B: L2 Regularization ---
print("\nTraining L2 Model...")
l2_rate = 0.001
model_l2 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,), 
                          kernel_regularizer=tf.keras.regularizers.l2(l2_rate)),
    tf.keras.layers.Dense(32, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(l2_rate)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_l2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
all_histories['L2'] = model_l2.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_scaled, y_val), verbose=verbose)
results['L2'] = evaluate_model(model_l2, X_test_scaled, y_test, "L2 Model")

# --- Strategy C: Dropout ---
print("\nTraining Dropout Model...")
drop_rate = 0.3
model_drop = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dropout(drop_rate), # Randomly drop 30% of neurons
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(drop_rate), # Randomly drop 30% of neurons
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_drop.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
all_histories['Dropout'] = model_drop.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_scaled, y_val), verbose=verbose)
results['Dropout'] = evaluate_model(model_drop, X_test_scaled, y_test, "Dropout Model")


# 3. Compare Results
plot_history(all_histories, "Regularization Comparison")

print("\n--- Summary of Test Set Performance ---")
summary_df = pd.DataFrame(results).T
print(summary_df)

print("\n--- Discussion ---")
print("1. Did the 'Base' model show a large gap between Train and Val accuracy (Overfitting)?")
print("2. Did L2 or Dropout close that gap?")
print("3. Which model had the best F1-Score on the test set? That's likely your winner.")