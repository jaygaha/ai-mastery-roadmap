import pandas as pd
import numpy as np

# Reload original data (or a fresh copy) to start clean for the workflow
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'Gender': ['Male', 'Female', 'MALE ', 'Female', 'Male', 'Female', 'male', 'Female', 'Male', 'Female', 'Male', 'Female', 'FEMALE', 'Male', 'Female'],
    'SeniorCitizen': [0, 0, 1, 'No', 0, 1, 0, 1, 0, 0, 1, 0, 0, 'Yes', 0],
    'Partner': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Dependents': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes'],
    'Tenure': [24, 1, 10, 50, 8, 72, 5, 30, 12, 6, -2, 45, 1, 60, 3], # Added a negative tenure
    'MonthlyCharges': [70.0, 29.85, 104.80, 80.0, 99.65, 110.0, np.nan, 75.0, 90.0, 60.0, 150.0, 25.0, 120.0, 10.0, 85.0], # Added a very low charge
    'TotalCharges': [1676.0, 29.85, np.nan, 3950.0, 800.0, 7933.0, 350.0, 2250.0, np.nan, 360.0, 1800.0, 1125.0, 120.0, 600.0, 255.0],
    'Contract': ['Month-to-month', 'Month-to-month', 'Month-to-month', 'Two year', 'Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Month-to-month', 'Month-to-month', 'Two year', 'one year', 'Month-to-month', 'Two year', 'Month-to-month'],
    'Churn': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', np.nan, 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check']
}
df_churn_cleaned = pd.DataFrame(data)

print("--- Initial DataFrame Snapshot ---")
df_churn_cleaned.info()
print("\nMissing values:\n", df_churn_cleaned.isnull().sum())
print("\nUnique values for categorical columns (before cleaning):")
for col in ['Gender', 'SeniorCitizen', 'Contract']:
    print(f"{col}: {df_churn_cleaned[col].unique()}")
print(f"Tenure min/max: {df_churn_cleaned['Tenure'].min()}/{df_churn_cleaned['Tenure'].max()}")
print("-" * 30)

# --- Step 1: Handle Missing Values ---
print("\n--- Step 1: Handling Missing Values ---")
# Identify columns with missing values
missing_cols = df_churn_cleaned.columns[df_churn_cleaned.isnull().any()].tolist()
print(f"Columns with missing values: {missing_cols}")

# Strategy: Impute numerical columns with median, categorical with mode
# Convert 'TotalCharges' to numeric first, coercing errors to NaN
# This is crucial because if 'TotalCharges' has an empty string or other non-numeric value,
# it won't be numeric and its mean/median calculation will fail or be incorrect.
df_churn_cleaned['TotalCharges'] = pd.to_numeric(df_churn_cleaned['TotalCharges'], errors='coerce')

# Impute 'MonthlyCharges' and 'TotalCharges' with their medians (robust to outliers)
for col in ['MonthlyCharges', 'TotalCharges']:
    if col in missing_cols:
        median_val = df_churn_cleaned[col].median()
        df_churn_cleaned[col].fillna(median_val, inplace=True)
        print(f"Imputed '{col}' with median: {median_val}")

# Impute 'PaymentMethod' (categorical) with mode
if 'PaymentMethod' in missing_cols:
    mode_val = df_churn_cleaned['PaymentMethod'].mode()[0]
    df_churn_cleaned['PaymentMethod'].fillna(mode_val, inplace=True)
    print(f"Imputed 'PaymentMethod' with mode: {mode_val}")

print("\nMissing values after imputation:\n", df_churn_cleaned.isnull().sum())
print("-" * 30)

# --- Step 2: Address Data Inconsistencies ---
print("\n--- Step 2: Addressing Data Inconsistencies ---")

# a) Data Type Conversion
# Convert 'SeniorCitizen' from potential strings ('Yes', 'No') to numeric (1, 0)
df_churn_cleaned['SeniorCitizen'] = df_churn_cleaned['SeniorCitizen'].replace({'Yes': 1, 'No': 0})
df_churn_cleaned['SeniorCitizen'] = pd.to_numeric(df_churn_cleaned['SeniorCitizen'], errors='coerce').astype(int)
print("Converted 'SeniorCitizen' to numeric (0/1).")

# b) Structural Errors (Case, Whitespace)
df_churn_cleaned['Gender'] = df_churn_cleaned['Gender'].str.strip().str.capitalize()
print("Standardized 'Gender' (strip, capitalize).")

df_churn_cleaned['Contract'] = df_churn_cleaned['Contract'].str.lower().str.capitalize() # Initial capitalization
# Then, specific replacements for consistency if needed (e.g., 'two year' to 'Two year')
df_churn_cleaned['Contract'] = df_churn_cleaned['Contract'].replace({
    'Two year': 'Two year', # Ensure exact match if needed
    'Month-to-month': 'Month-to-month',
    'One year': 'One year'
})
print("Standardized 'Contract' (strip, capitalize, specific replacements).")

# c) Logical Errors (Out-of-range values)
# Tenure cannot be negative. Replace negative values with 0.
df_churn_cleaned['Tenure'] = np.where(df_churn_cleaned['Tenure'] < 0, 0, df_churn_cleaned['Tenure'])
print("Corrected negative 'Tenure' values to 0.")

# MonthlyCharges cannot be less than 0. Replace negative values with 0.
df_churn_cleaned['MonthlyCharges'] = np.where(df_churn_cleaned['MonthlyCharges'] < 0, 0, df_churn_cleaned['MonthlyCharges'])
print("Corrected negative 'MonthlyCharges' values to 0.")

print("\nUnique values for categorical columns (after inconsistencies cleaning):")
for col in ['Gender', 'SeniorCitizen', 'Contract']:
    print(f"{col}: {df_churn_cleaned[col].unique()}")
print(f"Tenure min/max: {df_churn_cleaned['Tenure'].min()}/{df_churn_cleaned['Tenure'].max()}")
print("-" * 30)

# --- Step 3: Handle Outliers (Example for 'MonthlyCharges') ---
print("\n--- Step 3: Handling Outliers (MonthlyCharges) ---")
# Recalculate IQR bounds as data might have changed
Q1_mc = df_churn_cleaned['MonthlyCharges'].quantile(0.25)
Q3_mc = df_churn_cleaned['MonthlyCharges'].quantile(0.75)
IQR_mc = Q3_mc - Q1_mc
lower_bound_mc = Q1_mc - 1.5 * IQR_mc
upper_bound_mc = Q3_mc + 1.5 * IQR_mc

print(f"MonthlyCharges new IQR bounds: [{lower_bound_mc:.2f}, {upper_bound_mc:.2f}]")

# Capping outliers in 'MonthlyCharges'
df_churn_cleaned['MonthlyCharges'] = np.where(
    df_churn_cleaned['MonthlyCharges'] > upper_bound_mc,
    upper_bound_mc,
    np.where(
        df_churn_cleaned['MonthlyCharges'] < lower_bound_mc,
        lower_bound_mc,
        df_churn_cleaned['MonthlyCharges']
    )
)
print("Capped outliers in 'MonthlyCharges' using IQR bounds.")

print("\nFinal MonthlyCharges statistics after capping:")
print(df_churn_cleaned['MonthlyCharges'].describe())
print("-" * 30)

print("\n--- Final Cleaned DataFrame Snapshot ---")
df_churn_cleaned.info()
print("\nMissing values:\n", df_churn_cleaned.isnull().sum())
print("\nUnique values for categorical columns (final check):")
for col in ['Gender', 'SeniorCitizen', 'Contract']:
    print(f"{col}: {df_churn_cleaned[col].unique()}")
print(f"Tenure min/max: {df_churn_cleaned['Tenure'].min()}/{df_churn_cleaned['Tenure'].max()}")
print("\nFirst 5 rows of cleaned data:")
print(df_churn_cleaned.head())