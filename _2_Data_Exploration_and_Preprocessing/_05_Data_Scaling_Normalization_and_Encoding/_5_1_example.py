import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder

# Hypothetical DataFrame based on the Customer Churn case study
data = {
    'Customer ID': ['C1', 'C2', 'C3', 'C4', 'C5'],
    'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
    'TotalCharges': [29.85, 1889.50, 108.15, 1840.75, 151.65],
    'Age': [30, 45, 22, 60, 38],
    'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month'],
    'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
    'SeniorCitizen': ['No', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("-" * 30)

# 1. Data Scaling (Numerical Features)
# Identify numerical features for scaling
numerical_features = ['MonthlyCharges', 'TotalCharges', 'Age']

# Apply StandardScaler
scaler_standard = StandardScaler()
df_scaled_standard = df.copy() # Create a copy to avoid modifying original
df_scaled_standard[numerical_features] = scaler_standard.fit_transform(df[numerical_features])
print("DataFrame after StandardScaler (Z-score Normalization):")
print(df_scaled_standard[numerical_features].head())
print("-" * 30)

# Apply MinMaxScaler
scaler_minmax = MinMaxScaler()
df_scaled_minmax = df.copy() # Create another copy
df_scaled_minmax[numerical_features] = scaler_minmax.fit_transform(df[numerical_features])
print("DataFrame after MinMaxScaler (Normalization to [0, 1]):")
print(df_scaled_minmax[numerical_features].head())
print("-" * 30)

# 2. Encoding Categorical Variables

# Identify categorical features for encoding
categorical_nominal_features = ['Contract', 'Gender'] # No inherent order
categorical_ordinal_features = ['SeniorCitizen'] # Can be treated as ordinal or binary

# Apply One-Hot Encoding to 'Contract' and 'Gender'
encoder_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# Fit and transform on the selected features
encoded_features = encoder_ohe.fit_transform(df[categorical_nominal_features])

# Create a DataFrame from the encoded features with appropriate column names
encoded_df = pd.DataFrame(encoded_features, columns=encoder_ohe.get_feature_names_out(categorical_nominal_features))

# Concatenate the encoded DataFrame with the original DataFrame (dropping original columns)
df_encoded_ohe = pd.concat([df.drop(columns=categorical_nominal_features), encoded_df], axis=1)

print("DataFrame after One-Hot Encoding 'Contract' and 'Gender':")
print(df_encoded_ohe.head())
print("-" * 30)

# Apply Label Encoding to 'SeniorCitizen'
encoder_label = LabelEncoder()
df_encoded_label = df_encoded_ohe.copy() # Continue from the OHE DataFrame
df_encoded_label['SeniorCitizen'] = encoder_label.fit_transform(df_encoded_label['SeniorCitizen'])

print("DataFrame after Label Encoding 'SeniorCitizen':")
print(df_encoded_label.head())
print("-" * 30)

# Full preprocessed DataFrame combining scaling and encoding (example using StandardScaler)
# First, apply encoding to the original dataframe to get all features numerical
df_for_full_preprocessing = df.copy()

# One-hot encode nominal features
ohe_encoder_full = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe_transformed = ohe_encoder_full.fit_transform(df_for_full_preprocessing[categorical_nominal_features])
ohe_df_full = pd.DataFrame(ohe_transformed, columns=ohe_encoder_full.get_feature_names_out(categorical_nominal_features))
df_for_full_preprocessing = pd.concat([df_for_full_preprocessing.drop(columns=categorical_nominal_features), ohe_df_full], axis=1)

# Label encode ordinal features
label_encoder_full = LabelEncoder()
df_for_full_preprocessing['SeniorCitizen'] = label_encoder_full.fit_transform(df_for_full_preprocessing['SeniorCitizen'])

# Now apply scaling to all numerical features, including the newly encoded ones
# Identify all numerical features present after encoding
all_numerical_features = df_for_full_preprocessing.select_dtypes(include=['number']).columns.tolist()
# Exclude 'Customer ID' if it's not a feature for the model
if 'Customer ID' in all_numerical_features:
    all_numerical_features.remove('Customer ID')

# Apply StandardScaler to all numerical features
full_scaler = StandardScaler()
df_processed = df_for_full_preprocessing.copy()
df_processed[all_numerical_features] = full_scaler.fit_transform(df_for_full_preprocessing[all_numerical_features])

print("Fully Preprocessed DataFrame (after OHE, Label Encoding, and StandardScaler):")
print(df_processed.head())
print(df_processed.describe())