import pandas as pd
import numpy as np

# Create a sample DataFrame resembling the Customer Churn data
# with some deliberate missing values for demonstration
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'SeniorCitizen': [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'Partner': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No'],
    'Dependents': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Tenure': [24, 1, 10, 50, 8, 72, 5, 30, 12, 6],
    'MonthlyCharges': [70.0, 29.85, 104.80, 80.0, 99.65, 110.0, np.nan, 75.0, 90.0, 60.0],
    'TotalCharges': [1676.0, 29.85, np.nan, 3950.0, 800.0, 7933.0, 350.0, 2250.0, np.nan, 360.0],
    'Contract': ['Month-to-month', 'Month-to-month', 'Month-to-month', 'Two year', 'Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Month-to-month', 'Month-to-month'],
    'Churn': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}
df_churn = pd.DataFrame(data)

print("Initial DataFrame head:")
print(df_churn.head())
print("\nDataFrame Info:")
df_churn.info()

# Check for missing values across the entire DataFrame
print("\nMissing values in the DataFrame:")
print(df_churn.isnull())

# Count missing values per column
print("\nNumber of missing values per column:")
print(df_churn.isnull().sum())

# Calculate percentage of missing values per column
print("\nPercentage of missing values per column:")
print((df_churn.isnull().sum() / len(df_churn)) * 100)

# Strategies for Handling Missing Values

# Create a copy to demonstrate deletion without altering the original DataFrame
df_churn_dropped_rows = df_churn.copy()
print("\nDataFrame before dropping rows (shape):", df_churn_dropped_rows.shape)

# Drop rows with any missing values
df_churn_dropped_rows.dropna(inplace=True)
print("\nDataFrame after dropping rows with any missing values (shape):", df_churn_dropped_rows.shape)
print("Missing values after dropping rows:")
print(df_churn_dropped_rows.isnull().sum())

# Column-wise Deletion (dropna(axis=1))

# Create another copy for column dropping
df_churn_dropped_cols = df_churn.copy()
print("\nDataFrame before dropping columns (shape):", df_churn_dropped_cols.shape)

# Drop columns with any missing values
# We'll demonstrate this on a DataFrame where a column has many NaNs
# For our current small example, TotalCharges is 20% missing, which might be too much to drop in a real scenario
# Let's create a scenario where a column is mostly empty
df_churn_dropped_cols['NewFeature_ManyNaNs'] = [1, np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan]
print("\nDataFrame with an additional column 'NewFeature_ManyNaNs':")
print(df_churn_dropped_cols.isnull().sum())

df_churn_dropped_cols.dropna(axis=1, inplace=True)
print("\nDataFrame after dropping columns with any missing values (shape):", df_churn_dropped_cols.shape)
print("Columns after dropping:", df_churn_dropped_cols.columns.tolist())

# Imputation (Filling Missing Values)

# Resetting df_churn for imputation examples
df_churn_imputed = df_churn.copy()

print("Missing values before imputation:")
print(df_churn_imputed.isnull().sum())

# Impute 'MonthlyCharges' with the median
# The median is generally robust to outliers, making it a good choice for numerical data
median_monthly_charges = df_churn_imputed['MonthlyCharges'].median()
df_churn_imputed['MonthlyCharges'].fillna(median_monthly_charges, inplace=True)
print(f"\n'MonthlyCharges' imputed with median: {median_monthly_charges}")

# Impute 'TotalCharges' with the mean
# For demonstration, let's use mean here, though median might be safer for financial data if skewed.
mean_total_charges = df_churn_imputed['TotalCharges'].mean()
df_churn_imputed['TotalCharges'].fillna(mean_total_charges, inplace=True)
print(f"'TotalCharges' imputed with mean: {mean_total_charges}")

# In a real scenario, you might have categorical columns with missing values.
# Let's imagine a 'PaymentMethod' column that has a missing value
df_churn_imputed['PaymentMethod'] = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', np.nan, 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check']
print("\nMissing values in 'PaymentMethod' before imputation:")
print(df_churn_imputed['PaymentMethod'].isnull().sum())

# Impute 'PaymentMethod' with the mode
mode_payment_method = df_churn_imputed['PaymentMethod'].mode()[0] # .mode() can return multiple if tied, take first
df_churn_imputed['PaymentMethod'].fillna(mode_payment_method, inplace=True)
print(f"'PaymentMethod' imputed with mode: {mode_payment_method}")

print("\nMissing values after imputation:")
print(df_churn_imputed.isnull().sum())
print("\nDataFrame head after imputation:")
print(df_churn_imputed.head())

# Forward/Backward Fill (for sequential data)
# Create a small sample for ffill/bfill
df_sequential = pd.DataFrame({
    'Date': pd.to_datetime(['2025-11-01', '2025-11-02', '2025-11-03', '2025-11-04', '2025-11-05']),
    'Value': [10, np.nan, 12, np.nan, 15]
})
print("\nSequential data before ffill:")
print(df_sequential)

df_sequential['Value_ffill'] = df_sequential['Value'].ffill()
print("\nSequential data after ffill:")
print(df_sequential)

df_sequential['Value_bfill'] = df_sequential['Value'].bfill()
print("\nSequential data after bfill:")
print(df_sequential)


# Identifying Outliers
# 1. Visual Inspection
import matplotlib.pyplot as plt
import seaborn as sns

# Let's use a numerical column from our churn data, e.g., 'Tenure' or 'MonthlyCharges'
# First, ensure numerical columns are of the correct type (important after imputation)
df_churn_imputed['MonthlyCharges'] = pd.to_numeric(df_churn_imputed['MonthlyCharges'], errors='coerce')
df_churn_imputed['TotalCharges'] = pd.to_numeric(df_churn_imputed['TotalCharges'], errors='coerce')
df_churn_imputed['Tenure'] = pd.to_numeric(df_churn_imputed['Tenure'], errors='coerce')


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df_churn_imputed['MonthlyCharges'])
plt.title('Box Plot of MonthlyCharges')

plt.subplot(1, 2, 2)
sns.histplot(df_churn_imputed['MonthlyCharges'], kde=True)
plt.title('Histogram of MonthlyCharges')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df_churn_imputed['Tenure'])
plt.title('Box Plot of Tenure')

plt.subplot(1, 2, 2)
sns.histplot(df_churn_imputed['Tenure'], kde=True)
plt.title('Histogram of Tenure')

plt.tight_layout()
plt.show()

# For scatter plot, let's plot MonthlyCharges vs. TotalCharges
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Tenure', y='MonthlyCharges', data=df_churn_imputed)
plt.title('Scatter Plot of Tenure vs. MonthlyCharges')
plt.show()

# 2. Statistical Methods

# Using IQR method for 'MonthlyCharges'
Q1 = df_churn_imputed['MonthlyCharges'].quantile(0.25)
Q3 = df_churn_imputed['MonthlyCharges'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"MonthlyCharges - Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"MonthlyCharges - Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")

outliers_iqr = df_churn_imputed[(df_churn_imputed['MonthlyCharges'] < lower_bound) | (df_churn_imputed['MonthlyCharges'] > upper_bound)]
print("\nOutliers in 'MonthlyCharges' (IQR method):")
print(outliers_iqr[['CustomerID', 'MonthlyCharges']])

# Using Z-score method for 'MonthlyCharges'
from scipy.stats import zscore

# Ensure the column is numeric for zscore calculation
df_churn_imputed['MonthlyCharges_numeric'] = pd.to_numeric(df_churn_imputed['MonthlyCharges'], errors='coerce')
z_scores = np.abs(zscore(df_churn_imputed['MonthlyCharges_numeric'])) # Calculate absolute Z-scores
outliers_zscore = df_churn_imputed[z_scores > 2] # Threshold of 2 for demonstration (can be 3 or higher)
print("\nOutliers in 'MonthlyCharges' (Z-score method, threshold 2):")
print(outliers_zscore[['CustomerID', 'MonthlyCharges']])

# Treating Outliers

# 1. Deletion (Removing Outlier Rows)

df_churn_no_outliers_deleted = df_churn_imputed.copy()
print("Shape before outlier deletion:", df_churn_no_outliers_deleted.shape)

# Removing outliers identified by IQR method
df_churn_no_outliers_deleted = df_churn_no_outliers_deleted[
    (df_churn_no_outliers_deleted['MonthlyCharges'] >= lower_bound) &
    (df_churn_no_outliers_deleted['MonthlyCharges'] <= upper_bound)
]
print("Shape after outlier deletion (IQR method on MonthlyCharges):", df_churn_no_outliers_deleted.shape)

# 2. Capping (Winsorization)

df_churn_capped = df_churn_imputed.copy()

# Capping 'MonthlyCharges' using IQR bounds
df_churn_capped['MonthlyCharges'] = np.where(
    df_churn_capped['MonthlyCharges'] > upper_bound,
    upper_bound,
    np.where(
        df_churn_capped['MonthlyCharges'] < lower_bound,
        lower_bound,
        df_churn_capped['MonthlyCharges']
    )
)

print("\nOriginal MonthlyCharges statistics:")
print(df_churn_imputed['MonthlyCharges'].describe())
print("\nCapped MonthlyCharges statistics (using IQR bounds):")
print(df_churn_capped['MonthlyCharges'].describe())

# Verify if any values exceed the bounds after capping
print(f"Max MonthlyCharges after capping: {df_churn_capped['MonthlyCharges'].max():.2f}")
print(f"Min MonthlyCharges after capping: {df_churn_capped['MonthlyCharges'].min():.2f}")

# Visualize the effect of capping
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_churn_imputed['MonthlyCharges'])
plt.title('MonthlyCharges (Original)')
plt.subplot(1, 2, 2)
sns.boxplot(y=df_churn_capped['MonthlyCharges'])
plt.title('MonthlyCharges (Capped)')
plt.tight_layout()
plt.show()

# Addressing Data Inconsistencies
# Identifying Inconsistencies

# Let's ensure TotalCharges is numeric, as sometimes missing values can make it 'object' type
df_churn_imputed['TotalCharges'] = pd.to_numeric(df_churn_imputed['TotalCharges'], errors='coerce')
# Check info again
print("DataFrame info after ensuring TotalCharges is numeric:")
df_churn_imputed.info()

print("\nUnique values for 'Gender':")
print(df_churn_imputed['Gender'].unique())

print("\nValue counts for 'Contract':")
print(df_churn_imputed['Contract'].value_counts())

# Introduce an inconsistency for demonstration
df_churn_imputed.loc[0, 'Gender'] = 'MALE '
df_churn_imputed.loc[3, 'Contract'] = 'two year'
df_churn_imputed.loc[5, 'SeniorCitizen'] = 'Yes' # Mix type
df_churn_imputed.loc[8, 'Tenure'] = -5 # Logical error
df_churn_imputed.loc[1, 'MonthlyCharges'] = 1500 # A very high (but not necessarily outlier) value, just for type example

print("\nUnique values for 'Gender' after introducing inconsistency:")
print(df_churn_imputed['Gender'].unique())

print("\nValue counts for 'Contract' after introducing inconsistency:")
print(df_churn_imputed['Contract'].value_counts())

print("\nUnique values for 'SeniorCitizen' after introducing inconsistency:")
print(df_churn_imputed['SeniorCitizen'].unique())

print("\nMin/Max for 'Tenure' after introducing logical error:")
print(f"Min Tenure: {df_churn_imputed['Tenure'].min()}")
print(f"Max Tenure: {df_churn_imputed['Tenure'].max()}")

# Correcting Inconsistencies

# 1. Data Type Conversion

# Correcting 'SeniorCitizen' to numeric (0 or 1)
# First, map 'Yes' to 1 and 'No' to 0 for consistency, then convert to int
df_churn_imputed['SeniorCitizen'] = df_churn_imputed['SeniorCitizen'].replace({'Yes': 1, 'No': 0})
df_churn_imputed['SeniorCitizen'] = pd.to_numeric(df_churn_imputed['SeniorCitizen'], errors='coerce').astype(int)

# Check the data type and unique values again
print("\nUnique values for 'SeniorCitizen' after conversion:")
print(df_churn_imputed['SeniorCitizen'].unique())
print("Data type of 'SeniorCitizen':", df_churn_imputed['SeniorCitizen'].dtype)

# 2. Structural and Format Errors
# Standardize 'Gender'
df_churn_imputed['Gender'] = df_churn_imputed['Gender'].str.strip().str.capitalize()
print("\nUnique values for 'Gender' after standardization:")
print(df_churn_imputed['Gender'].unique())

# Standardize 'Contract'
df_churn_imputed['Contract'] = df_churn_imputed['Contract'].str.lower().str.replace('two year', 'Two year').str.replace('month-to-month', 'Month-to-month').str.replace('one year', 'One year') # Example of fixing specific cases
print("\nValue counts for 'Contract' after standardization:")
print(df_churn_imputed['Contract'].value_counts())

# Logical Errors (Values outside valid range)

# Correcting 'Tenure' logical error (negative tenure)
# Tenure cannot be negative. We might replace it with 0 (new customer), median, or mark it as NaN for further handling.
# Let's replace negative tenure with 0 assuming it indicates a very new customer or an error that should be minimal.
df_churn_imputed['Tenure'] = np.where(df_churn_imputed['Tenure'] < 0, 0, df_churn_imputed['Tenure'])
print("\nMin/Max for 'Tenure' after correcting logical error:")
print(f"Min Tenure: {df_churn_imputed['Tenure'].min()}")
print(f"Max Tenure: {df_churn_imputed['Tenure'].max()}")

# Example for MonthlyCharges: Assuming charges cannot be less than 0
df_churn_imputed['MonthlyCharges'] = np.where(df_churn_imputed['MonthlyCharges'] < 0, 0, df_churn_imputed['MonthlyCharges'])
print("\nMin MonthlyCharges after correcting logical error (if any):")
print(f"Min MonthlyCharges: {df_churn_imputed['MonthlyCharges'].min()}")