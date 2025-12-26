import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

"""
Applying Random Forest for Customer Churn Prediction

This script demonstrates how to use the Random Forest algorithm to predict customer churn.
Random Forest is an ensemble learning method that constructs multiple decision trees during training
and outputs the class that is the mode of the classes (classification) of the individual trees.

Key Steps:
1. Load Data
2. Preprocess Data (Cleaning, Encoding, Scaling)
3. Split Data into Training and Testing Sets
4. Initialize and Train Random Forest Model
5. make Predictions
6. Evaluate Model Performance
7. Analyze Feature Importance
"""

# --- 1. Load Data and Initial Preprocessing ---
# (Using the robust preprocessing pipeline established in previous modules)

print("--- 1. Loading and Preprocessing Data ---")

# Load the dataset
try:
    df = pd.read_csv('../../_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling/Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Dataset not found. Please ensure 'Telco-Customer-Churn.csv' is in the correct path.")
    # Fallback for local execution if running directly in the directory
    try:
        df = pd.read_csv('Telco-Customer-Churn.csv')
    except FileNotFoundError:
        print("Could not find the dataset. Exiting...")
        exit()

# Initial Data Cleaning and Feature Engineering
# Convert 'TotalCharges' to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Fill missing 'TotalCharges' with the median
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
# Drop 'customerID' as it's not predictive
df.drop('customerID', axis=1, inplace=True)
# Map 'Churn' to binary values (1 for Yes, 0 for No)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Simplify service columns
service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})

# Create a new feature 'NumServices'
df['NumServices'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# --- 2. Train-Test Split ---
# Split the data into training and testing sets (75% train, 25% test)
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- 3. Preprocessing Setup ---
# Configure the column transformer for scaling and encoding
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
nominal_categorical_cols = [col for col in X_train.select_dtypes(include=['object']).columns.tolist() if col != 'Contract']
contract_order = [['Month-to-month', 'One year', 'Two year']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat_nominal', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_categorical_cols),
        ('cat_ordinal', OrdinalEncoder(categories=contract_order), ['Contract'])
    ],
    remainder='passthrough'
)

# --- 4. Apply Preprocessing ---
print("Applying preprocessing transformations...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to dense arrays if necessary (not strictly required for RF but good for consistency)
if hasattr(X_train_processed, 'toarray'):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_test_processed, 'toarray'):
    X_test_processed = X_test_processed.toarray()

# --- 5. Initialize and Train Random Forest Model ---
print("\n--- 2. Training Random Forest Model ---")
# Initialize the Random Forest Classifier
# n_estimators: The number of trees in the forest (100 is a good starting point).
# max_depth: The maximum depth of the tree. Limiting this helps prevent overfitting.
# random_state: Controls the randomness of the bootstrapping of the samples used when building trees.
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model
rf_model.fit(X_train_processed, y_train)
print("Random Forest Model Trained Successfully.")

# --- 6. Make Predictions ---
print("Making predictions on the test set...")
y_pred_rf = rf_model.predict(X_test_processed)
# Get probabilities for the positive class (Churn)
y_pred_proba_rf = rf_model.predict_proba(X_test_processed)[:, 1]

# --- 7. Evaluate Model Performance ---
print("\n--- 3. Model Evaluation ---")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest Precision: {precision_rf:.4f}")
print(f"Random Forest Recall: {recall_rf:.4f}")
print(f"Random Forest F1-Score: {f1_rf:.4f}")
print("\nRandom Forest Confusion Matrix:")
print(conf_matrix_rf)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Predicted No Churn', 'Predicted Churn'],
            yticklabels=['Actual No Churn', 'Actual Churn'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# --- 8. Feature Importance ---
# One of the great benefits of Random Forests is easy access to feature importance
print("\n--- 4. Feature Importance Analysis ---")

# Get feature names from the preprocessor
feature_names_out = preprocessor.get_feature_names_out()

# Create a Series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(rf_model.feature_importances_, index=feature_names_out).sort_values(ascending=False)

print("Top 10 Most Important Features:")
print(feature_importances.head(10))

# Visualize Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.head(10), y=feature_importances.head(10).index, hue=feature_importances.head(10).index, palette='viridis', legend=False)
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

print("\n--- Interpretation ---")
print("Top features indicate which variables have the most influence on defining customer churn.")
