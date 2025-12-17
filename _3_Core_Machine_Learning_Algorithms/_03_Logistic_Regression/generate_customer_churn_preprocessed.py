import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Simulate raw customer churn data
np.random.seed(42)
num_customers = 1000

data = {
    'customerID': [f'CUST_{i:04d}' for i in range(num_customers)],
    'gender': np.random.choice(['Male', 'Female'], num_customers),
    'SeniorCitizen': np.random.choice([0, 1], num_customers, p=[0.8, 0.2]),
    'Partner': np.random.choice(['Yes', 'No'], num_customers),
    'Dependents': np.random.choice(['Yes', 'No'], num_customers),
    'tenure': np.random.randint(1, 73, num_customers),
    'PhoneService': np.random.choice(['Yes', 'No'], num_customers),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], num_customers),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], num_customers, p=[0.35, 0.45, 0.20]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], num_customers),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], num_customers),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], num_customers),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], num_customers),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], num_customers),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], num_customers),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_customers, p=[0.5, 0.25, 0.25]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], num_customers),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], num_customers),
    'MonthlyCharges': np.random.uniform(18.0, 118.75, num_customers),
    'TotalCharges': np.random.uniform(20.0, 8684.8, num_customers),
    'Churn': np.random.choice(['Yes', 'No'], num_customers, p=[0.26, 0.74]) # Approx churn rate
}

df = pd.DataFrame(data)

# Introduce some missing values for 'TotalCharges' to simulate real-world data
missing_indices = np.random.choice(df.index, 20, replace=False)
df.loc[missing_indices, 'TotalCharges'] = np.nan

# Handle 'No internet service' and 'No phone service' consistently
for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
    df[col] = df[col].replace('No internet service', 'No')
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

# Convert 'TotalCharges' to numeric, coercing errors will turn non-numeric (like empty strings) into NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing 'TotalCharges' with the median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID as it's not a feature
df = df.drop('customerID', axis=1)

# Convert 'Yes'/'No' to 1/0 for binary categorical features
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# For 'gender', map 'Female' to 1 and 'Male' to 0
df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# Define categorical and numerical features for preprocessing
categorical_features = df.select_dtypes(include='object').columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn') # Exclude target

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (like Churn) as is
)

# Apply preprocessing
# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Fit and transform the data
X_preprocessed_array = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(numerical_features) + list(onehot_feature_names)

# Create a DataFrame with preprocessed features
X_preprocessed = pd.DataFrame(X_preprocessed_array, columns=all_feature_names)

# Combine preprocessed features with the target variable
df_preprocessed = pd.concat([X_preprocessed, y.reset_index(drop=True)], axis=1)

# Save the preprocessed DataFrame to a CSV file
df_preprocessed.to_csv('customer_churn_preprocessed.csv', index=False)

print("customer_churn_preprocessed.csv generated successfully.")
print(df_preprocessed.head())
print(df_preprocessed.shape)