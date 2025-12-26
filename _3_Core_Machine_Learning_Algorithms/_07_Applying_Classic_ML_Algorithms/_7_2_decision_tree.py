import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree # Import DecisionTreeClassifier and plot_tree for visualization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns # For better looking plots

# --- 1. Load Data and Initial Preprocessing ---
# (Using the corrected and robust preprocessing pipeline from previous steps)

# Load the dataset
try:
    df = pd.read_csv('../../_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling/Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Dataset not found. Please ensure 'Telco-Customer-Churn.csv' is in the correct path.")
    # Exit or handle the error appropriately, perhaps by trying a different path
    # For this example, we'll assume it's found for execution
    df = pd.read_csv('Telco-Customer-Churn.csv') # Fallback for local execution if running directly

# Initial Data Cleaning and Feature Engineering
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})

df['NumServices'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# --- 2. Train-Test Split ---
# This is the ONLY place where train_test_split should occur.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- 3. Preprocessing Setup ---
# Identify column types based on X_train
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
nominal_categorical_cols = [col for col in X_train.select_dtypes(include=['object']).columns.tolist() if col != 'Contract']
contract_order = [['Month-to-month', 'One year', 'Two year']] # Define the order for 'Contract'

# Create the preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat_nominal', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_categorical_cols),
        ('cat_ordinal', OrdinalEncoder(categories=contract_order), ['Contract'])
    ],
    remainder='passthrough'
)

# --- 4. Apply Preprocessing ---
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to dense arrays (ColumnTransformer often outputs sparse matrices)
if hasattr(X_train_processed, 'toarray'):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_test_processed, 'toarray'):
    X_test_processed = X_test_processed.toarray()

# --- 5. Initialize and Train Decision Tree Classifier ---
print("\n--- Training Decision Tree Classifier ---")
# Instantiate the Decision Tree Classifier
# Key hyperparameters:
# - max_depth: Limits the depth of the tree to prevent overfitting.
# - min_samples_leaf: Minimum number of samples required to be at a leaf node.
# - criterion: Function to measure the quality of a split (e.g., 'gini' for Gini impurity, 'entropy' for information gain).
dt_classifier = DecisionTreeClassifier(
    max_depth=5,            # Limit tree depth to improve interpretability and reduce overfitting
    min_samples_leaf=50,    # Require at least 50 samples in a leaf node
    random_state=42,        # For reproducibility
    criterion='gini'        # Use Gini impurity
)

# Train the model
dt_classifier.fit(X_train_processed, y_train)
print("Decision Tree Model Trained Successfully.")

# --- 6. Make Predictions ---
y_pred_dt = dt_classifier.predict(X_test_processed)
y_pred_proba_dt = dt_classifier.predict_proba(X_test_processed)[:, 1] # Probability of the positive class (churn)

# --- 7. Evaluate Model Performance ---
print("\n--- Decision Tree Model Evaluation ---")
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
print(f"Decision Tree Precision: {precision_dt:.4f}")
print(f"Decision Tree Recall: {recall_dt:.4f}")
print(f"Decision Tree F1-Score: {f1_dt:.4f}")
print("\nDecision Tree Confusion Matrix:")
print(conf_matrix_dt)

# Visualizing the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted No Churn', 'Predicted Churn'],
            yticklabels=['Actual No Churn', 'Actual Churn'])
plt.title('Decision Tree Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# --- 8. Visualize the Decision Tree (Optional but highly recommended) ---
print("\n--- Visualizing the Decision Tree ---")

# Get feature names after preprocessing for better interpretability
feature_names_out = preprocessor.get_feature_names_out()

plt.figure(figsize=(25, 15)) # Adjust figure size for better readability
plot_tree(dt_classifier,
          feature_names=feature_names_out, # Use processed feature names
          class_names=['No Churn', 'Churn'], # Class labels
          filled=True, # Color nodes to indicate majority class
          rounded=True, # Round node boxes
          fontsize=10, # Adjust font size as needed
          proportion=False, # Show counts rather than proportions
          max_depth=3 # Only show first 3 levels for readability in plot
         )
plt.title("Decision Tree Visualization (First 3 Levels)")
plt.show()

print("\n--- Decision Tree Feature Importances ---")
# Feature importances sum to 1 and indicate how much each feature contributed to splits
feature_importances = pd.Series(dt_classifier.feature_importances_, index=feature_names_out)
print(feature_importances.sort_values(ascending=False))

print("\n--- Interpretation Guidance ---")
print("• Look at the decision path from the root to a leaf node to understand a prediction.")
print("• Features with higher importance are more influential in the decision-making process.")
print("• Observe how max_depth and min_samples_leaf impact the tree's complexity and performance.")
print("  - A deeper tree (higher max_depth, lower min_samples_leaf) can overfit but capture more detail.")
print("  - A shallower tree (lower max_depth, higher min_samples_leaf) is more generalized but might underfit.")