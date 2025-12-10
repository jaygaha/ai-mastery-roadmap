"""
Exercise

1. Experiment with Min-Max Scaling: Modify the provided code to use MinMaxScaler instead of StandardScaler for the numerical features. Observe any changes in the values of 
X_train_processed and X_test_processed. Explain when MinMaxScaler might be preferred over StandardScaler based on our discussion.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
try:
    df = pd.read_csv('Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Dataset not found. Please ensure 'Telco-Customer-Churn.csv' is in the correct path.")
    exit()

# Initial Data Cleaning and Feature Engineering
# Convert 'TotalCharges' to numeric. Coerce errors will turn non-numeric values (like ' ') into NaN.
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("="*80)
print("EXERCISE 1: MIN-MAX SCALING VS STANDARD SCALING")
print("="*80)

# Identify column types
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
nominal_categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Original: StandardScaler
print("\nOriginal: StandardScaler")
numerical_transformer_std = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor_std = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_std, numerical_cols),
        ('cat', categorical_transformer, nominal_categorical_cols)
    ])

X_train_std = preprocessor_std.fit_transform(X_train)
X_test_std = preprocessor_std.transform(X_test)


if hasattr(X_train_std, 'toarray'):
    X_train_std = X_train_std.toarray()
if hasattr(X_test_std, 'toarray'):
    X_test_std = X_test_std.toarray()

print(f"Shape: {X_train_std.shape}")
print(f"First 5 numerical values (first row): {X_train_std[0, :len(numerical_cols)]}")
print(f"Mean of first numerical feature: {X_train_std[:, 0].mean():.4f}")
print(f"Std of first numerical feature: {X_train_std[:, 0].std():.4f}")

# Modified: MinMaxScaler
print("\n--- Using MinMaxScaler (Modified) ---")
numerical_transformer_minmax = MinMaxScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor_minmax = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_minmax, numerical_cols),
        ('cat', categorical_transformer, nominal_categorical_cols)])

X_train_minmax = preprocessor_minmax.fit_transform(X_train)
X_test_minmax = preprocessor_minmax.transform(X_test)

if hasattr(X_train_minmax, 'toarray'):
    X_train_minmax = X_train_minmax.toarray()
if hasattr(X_test_minmax, 'toarray'):
    X_test_minmax = X_test_minmax.toarray()

print(f"Shape: {X_train_minmax.shape}")
print(f"First 5 numerical values (first row): {X_train_minmax[0, :len(numerical_cols)]}")
print(f"Min of first numerical feature: {X_train_minmax[:, 0].min():.4f}")
print(f"Max of first numerical feature: {X_train_minmax[:, 0].max():.4f}")

print("\n--- Comparison and Explanation ---")
print("StandardScaler: Transforms features to have mean=0 and std=1 (z-score normalization)")
print("MinMaxScaler: Transforms features to range [0, 1] (or specified range)")
print("\nWhen to prefer MinMaxScaler:")
print("• When you need bounded values (e.g., [0,1] for neural networks)")
print("• When features have a bounded distribution (not too many outliers)")
print("• When you want to preserve zero values")
print("• For algorithms sensitive to feature magnitude (e.g., neural networks, k-NN)")
print("\nWhen to prefer StandardScaler:")
print("• When data has Gaussian distribution")
print("• When there are outliers (MinMaxScaler is sensitive to outliers)")
print("• For algorithms assuming normally distributed features (e.g., Linear/Logistic Regression)")

"""
2. Ordinal Encoding for 'Contract': The Contract column has values 'Month-to-month', 'One year', 'Two year'. This column inherently has an order. Implement OrdinalEncoder for the 
Contract column instead of treating it as a nominal categorical variable with OneHotEncoder. You will need to define the order of categories manually for OrdinalEncoder to ensure correct 
ranking. Compare the shape of the processed data before and after this change.
"""

print("\n" + "="*80)
print("EXERCISE 2: ORDINAL ENCODING FOR 'CONTRACT'")
print("="*80)

# Separate Contract from other categorical columns
nominal_categorical_cols_ex2 = [col for col in nominal_categorical_cols if col != 'Contract']

print("\n--- Original: OneHotEncoder for all categorical (including Contract) ---")
preprocessor_original = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_categorical_cols)])

X_train_original = preprocessor_original.fit_transform(X_train)
if hasattr(X_train_original, 'toarray'):
    X_train_original = X_train_original.toarray()
print(f"Shape with OHE for Contract: {X_train_original.shape}")

# Modified: OrdinalEncoder for Contract
print("\n--- Modified: OrdinalEncoder for Contract ---")
# Define the order: Month-to-month < One year < Two year
contract_order = [['Month-to-month', 'One year', 'Two year']]

preprocessor_ordinal = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat_nominal', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_categorical_cols_ex2),
        ('cat_ordinal', OrdinalEncoder(categories=contract_order), ['Contract'])])

X_train_ordinal = preprocessor_ordinal.fit_transform(X_train)
X_test_ordinal = preprocessor_ordinal.transform(X_test)

if hasattr(X_train_ordinal, 'toarray'):
    X_train_ordinal = X_train_ordinal.toarray()
if hasattr(X_test_ordinal, 'toarray'):
    X_test_ordinal = X_test_ordinal.toarray()

print(f"Shape with Ordinal for Contract: {X_train_ordinal.shape}")
print(f"\nContract encoding mapping:")
print("  Month-to-month → 0")
print("  One year → 1")
print("  Two year → 2")
print(f"\nSample Contract values (encoded): {X_train_ordinal[:5, -1]}")

print("\n--- Shape Comparison ---")
print(f"Original (OHE for all): {X_train_original.shape}")
print(f"Modified (Ordinal for Contract): {X_train_ordinal.shape}")
print(f"Difference in features: {X_train_original.shape[1] - X_train_ordinal.shape[1]}")
print("\nExplanation:")
print("• OneHotEncoder creates k-1=2 columns for Contract (with drop='first')")
print("• OrdinalEncoder creates 1 column for Contract")
print("• Net reduction: 2-1 = 1 feature")
print("• Ordinal encoding preserves the inherent order and reduces dimensionality")

"""
3. Impact of drop='first': Remove drop='first' from the OneHotEncoder in the provided code. Run the code and observe the change in the shape of X_train_processed and X_test_processed. 
Explain why this change occurs and discuss the potential implications for model training (referencing the dummy variable trap).
"""

print("\n" + "="*80)
print("EXERCISE 3: IMPACT OF drop='first'")
print("="*80)

print("\n--- With drop='first' (Original - Avoids Dummy Variable Trap) ---")
preprocessor_drop = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_categorical_cols)])

X_train_drop = preprocessor_drop.fit_transform(X_train)
if hasattr(X_train_drop, 'toarray'):
    X_train_drop = X_train_drop.toarray()
print(f"Shape: {X_train_drop.shape}")

print("\n--- Without drop='first' (Modified - Creates Dummy Variable Trap) ---")
preprocessor_no_drop = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop=None), nominal_categorical_cols)])

X_train_no_drop = preprocessor_no_drop.fit_transform(X_train)
if hasattr(X_train_no_drop, 'toarray'):
    X_train_no_drop = X_train_no_drop.toarray()
print(f"Shape: {X_train_no_drop.shape}")

print("\n--- Analysis ---")
print(f"Difference in features: {X_train_no_drop.shape[1] - X_train_drop.shape[1]}")

# Count categorical features
cat_feature_count = len(nominal_categorical_cols)
print(f"\nNumber of categorical columns: {cat_feature_count}")
print(f"Additional features without drop='first': {X_train_no_drop.shape[1] - X_train_drop.shape[1]}")
print("(One extra dummy variable per categorical column)")

print("\n--- Dummy Variable Trap Explanation ---")
print("What is the Dummy Variable Trap?")
print("• When k categories are encoded into k binary columns instead of k-1")
print("• Creates perfect multicollinearity: one column is perfectly predictable from others")
print("• Example: gender with 'Male' and 'Female'")
print("  - Without drop: Male=1,Female=0 OR Male=0,Female=1")
print("  - With drop='first': Only Female=0 or 1 (Male is implicit)")
print("\nImplications for Model Training:")
print("• Linear models (Regression, Logistic Regression): Cannot compute unique coefficients")
print("  - Matrix becomes singular (non-invertible)")
print("  - Some libraries handle it, but results are unstable")
print("• Tree-based models (Random Forest, XGBoost): Less affected but wastes resources")
print("  - Redundant features don't hurt performance much")
print("  - Increases memory and computation without adding information")
print("• Neural Networks: Can handle it but may slow convergence")
print("\nBest Practice:")
print("• Always use drop='first' or drop='if_binary' for linear models")
print("• Optionally use it for tree-based models to reduce dimensionality")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Exercise 1: MinMaxScaler scales to [0,1], preferred for bounded distributions")
print("Exercise 2: OrdinalEncoder reduces dimensions while preserving order")
print("Exercise 3: drop='first' prevents multicollinearity (dummy variable trap)")
print("="*80)