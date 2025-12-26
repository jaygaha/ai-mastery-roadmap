import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('../../_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling/Telco-Customer-Churn.csv')

# Initial Data Cleaning and Feature Engineering (as done in your previous lessons)
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

# --- ONLY ONE TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
# Moved test_size=0.25 here, and ensure this is the ONLY split

# Identify column types based on X_train (correct practice)
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
nominal_categorical_cols = [col for col in X_train.select_dtypes(include=['object']).columns.tolist() if col != 'Contract']

# Define the order for the 'Contract' column
contract_order = [['Month-to-month', 'One year', 'Two year']]

# Create the preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat_nominal', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_categorical_cols),
        ('cat_ordinal', OrdinalEncoder(categories=contract_order), ['Contract'])
    ],
    remainder='passthrough'
)

# Fit preprocessor on X_train and transform both X_train and X_test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to dense arrays if sparse
if hasattr(X_train_processed, 'toarray'):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_test_processed, 'toarray'):
    X_test_processed = X_test_processed.toarray()

# Now X_train_processed, X_test_processed, y_train, y_test are all aligned and ready.

# Initialize and train the Logistic Regression model using the PROCESSED data
logistic_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000) # Added max_iter for convergence
logistic_model.fit(X_train_processed, y_train)

# Make predictions on the PROCESSED test set
y_pred_logistic = logistic_model.predict(X_test_processed)
y_pred_proba_logistic = logistic_model.predict_proba(X_test_processed)[:, 1]

# Evaluate the model with y_test from the original split
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic)
recall_logistic = recall_score(y_test, y_pred_logistic)
f1_logistic = f1_score(y_test, y_pred_logistic)
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)

print(f"Logistic Regression Accuracy: {accuracy_logistic:.4f}")
print(f"Logistic Regression Precision: {precision_logistic:.4f}")
print(f"Logistic Regression Recall: {recall_logistic:.4f}")
print(f"Logistic Regression F1-Score: {f1_logistic:.4f}")
print("Logistic Regression Confusion Matrix:")
print(conf_matrix_logistic)

# Example interpretation of coefficients
# To correctly map coefficients back, you need the feature names after preprocessing.
# This requires extracting them from the preprocessor.
# For simplicity, we'll iterate through the number of features.
# A more robust solution would involve getting feature names from OneHotEncoder.
print("\nLogistic Regression Coefficients (interpretation order might differ from original X.columns):")
# You'd need to reconstruct feature names after preprocessing to map these correctly.
# For numerical features, order is preserved. For OHE features, it gets complex.
# This loop will print coefficients, but the 'feature' names won't directly match the X.columns
# for the OHE parts, as OHE creates new columns.
feature_names_out = preprocessor.get_feature_names_out()
for feature, coef in zip(feature_names_out, logistic_model.coef_[0]):
    print(f"Feature: {feature}, Coefficient: {coef:.4f}")