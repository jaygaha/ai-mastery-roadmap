"""
2. Experiment with Activation Functions: Using the original model for binary classification, change the activation function in the hidden layers from 'relu' to 'sigmoid' or 'tanh' (hyperbolic tangent).

    - Retrain the model and observe the changes in performance (accuracy and loss).
    - Reflect on why relu is generally preferred for hidden layers in modern deep learning, especially concerning the vanishing gradient problem discussed in Module 4.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reload preprocessed data (assuming the same setup as Exercise 1)
try:
    df = pd.read_csv('../../_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling/Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Error: 'Telco-Customer-Churn.csv' not found. Please ensure your data is ready.")
    exit()

# Preprocessing
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)
    
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

if df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = tf.constant(X_train_scaled, dtype=tf.float32)
X_test_tensor = tf.constant(X_test_scaled, dtype=tf.float32)
y_train_tensor = tf.constant(y_train.values, dtype=tf.float32)
y_test_tensor = tf.constant(y_test.values, dtype=tf.float32)

input_dim = X_train_scaled.shape[1]
# Choose a fixed architecture for consistency (e.g., 1 hidden layer, 64 neurons)
HIDDEN_LAYERS = 1
NEURONS_PER_LAYER = [64] # List of neurons for each hidden layer

# Function to build and train a model with a specified activation
def train_and_evaluate_model(hidden_activation, model_name):
    print(f"\n--- Training Model with {model_name} Activation ---")
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for i, neurons in enumerate(NEURONS_PER_LAYER):
        model.add(layers.Dense(neurons, activation=hidden_activation, name=f'hidden_layer_{i+1}'))
    model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train_tensor, y_train_tensor,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.1,
                        verbose=0) # Set to 1 to see epoch-by-epoch progress

    y_pred_probs = model.predict(X_test_tensor)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print(f"{model_name} Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"{model_name} Model Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"{model_name} Model Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"{model_name} Model F1-Score: {f1_score(y_test, y_pred):.4f}")
    return model, history, {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

results = {}

# --- 1. Baseline Model (ReLU) ---
model_relu, history_relu, results['relu'] = train_and_evaluate_model('relu', 'ReLU')

# --- 2. Experiment with tanh Activation ---
model_tanh, history_tanh, results['tanh'] = train_and_evaluate_model('tanh', 'Tanh')

# --- 3. Experiment with sigmoid (as hidden layer activation) ---
model_sigmoid_hidden, history_sigmoid_hidden, results['sigmoid_hidden'] = train_and_evaluate_model('sigmoid', 'Sigmoid Hidden')

print("\n--- Comparative Analysis ---")
for activation_name, metrics in results.items():
    print(f"\n{activation_name} Model Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")

print("\n--- Discussion Points ---")
print("1. Which activation function (ReLU, Tanh, Sigmoid) yielded the best performance on your test set?")
print("2. Why do you think ReLU is often preferred in hidden layers for deep networks compared to Sigmoid or Tanh?")
print("3. What is the 'vanishing gradient' problem, and which activation functions are more susceptible to it?")
print("4. Did you notice differences in how quickly each model converged during training (e.g., looking at validation loss/accuracy curves if plotted, or just epoch-by-epoch training output if verbose=1)?")
