import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset (assuming 'Telco-Customer-Churn.csv' is in the same directory)
try:
    df = pd.read_csv('Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Dataset not found. Please ensure 'Telco-Customer-Churn.csv' is in the correct path.")
    exit()

# --- Step 1: Initial Data Cleaning and Feature Engineering (as discussed in previous lessons) ---
# Convert 'TotalCharges' to numeric. Coerce errors will turn non-numeric values (like ' ') into NaN.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Handle missing 'TotalCharges' by imputing with the median
# This handles cases where customers might have just started and have no total charges yet
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)

# Drop 'customerID' as it's an identifier and not a predictive feature
df.drop('customerID', axis=1, inplace=True)

# Convert target variable 'Churn' to numerical (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Engineer a new feature 'HasMultipleServices'
# Assuming 'No phone service' and 'No internet service' also count as 'No' for individual services
service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})

df['NumServices'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)

# Ensure 'gender' column is handled for consistency if it was an issue during EDA
# (e.g., if there were values other than 'Male'/'Female') - for this dataset, it's typically clean.

# --- Step 2: Separate features (X) and target (y) ---
X = df.drop('Churn', axis=1)
y = df['Churn']

# --- Step 3: Split the data into training and testing sets ---
# This is crucial to evaluate the model on unseen data.
# We use stratify=y to maintain the same proportion of churned/non-churned customers in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 4: Identify column types for preprocessing ---
# Numerical columns to be scaled
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Categorical columns to be One-Hot Encoded (nominal)
# Exclude 'gender' if we decide to drop one category to avoid dummy variable trap later, or handle it as binary
nominal_categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Ordinal categorical columns (none explicitly in this dataset for direct ordinal encoding,
# but we'll demonstrate for 'Contract' if we wanted a specific order, otherwise treat as nominal)
# For 'Contract' specifically, if we want to treat it as ordinal:
# ordinal_categorical_cols = ['Contract']
# For simplicity, we'll treat 'Contract' as nominal for this example.

# Remove 'gender' from nominal_categorical_cols for now and handle it separately if desired,
# or let OneHotEncoder handle it directly. For this example, we'll let OHE handle all nominals.
# However, if 'gender' only has 'Male'/'Female', we could manually map it (0/1) or let OHE create 2 columns.
# Let's check for 'gender' specifically:
# if 'gender' in nominal_categorical_cols:
#     nominal_categorical_cols.remove('gender') # If we were to encode it manually e.g. df['gender'].map({'Female':0, 'Male':1})

# Example: Define which columns will get which transformation
# For simplicity, let's treat 'gender' as a regular nominal categorical variable.

# Identify which numerical columns need scaling (exclude the 'NumServices' if it's already an integer count
# and you prefer not to scale it, though typically all numeric features are scaled).
# Let's scale all numerical columns.
print(f"Numerical columns to scale: {numerical_cols}")
print(f"Categorical columns to one-hot encode: {nominal_categorical_cols}")

# --- Step 5: Create preprocessing pipelines for numerical and categorical features ---
# Numerical transformer: Standard Scaler for our numerical columns
numerical_transformer = StandardScaler() # Or MinMaxScaler()

# Categorical transformer: One-Hot Encoder for our categorical columns
# handle_unknown='ignore' prevents errors during transformation if a category is seen in test data but not train.
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first') # drop='first' to avoid dummy variable trap

# Create a ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, nominal_categorical_cols)])

# --- Step 6: Create the full preprocessing and modeling pipeline ---
# At this stage, we're just building the preprocessing part.
# The full pipeline will include a model, which we'll cover in Module 3.
# We fit the preprocessor on the training data and then transform both training and testing data.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# The output of ColumnTransformer is a sparse matrix if any transformer returns one (like OneHotEncoder).
# We convert it to a dense array for most ML models.
if hasattr(X_train_processed, 'toarray'):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_test_processed, 'toarray'):
    X_test_processed = X_test_processed.toarray()

print("\nShape of processed training data:", X_train_processed.shape)
print("Shape of processed test data:", X_test_processed.shape)
print("\nFirst 5 rows of processed training data (numerical part and encoded categories):")
print(X_train_processed[:5])

# Get feature names after one-hot encoding for later analysis (optional, for interpretability)
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(nominal_categorical_cols)
all_feature_names = numerical_cols + list(ohe_feature_names)
print("\nExample of final feature names after preprocessing:")
print(all_feature_names[:10]) # Print first 10 for brevity