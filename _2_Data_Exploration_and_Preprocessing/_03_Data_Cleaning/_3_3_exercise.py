"""
Exercises and Practice Activities

To solidify your understanding of data cleaning, work through the following exercises using a new, slightly different version of the Customer Churn data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a new DataFrame for exercises
exercise_data = {
    'CustID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    'Age': [35, 22, 58, 41, np.nan, 65, 30, 48, 25, 150, 40, -5, 70, 33, 50],
    'Gender': ['male', 'Female', 'MALE', 'Female', 'MALE', 'Female', 'male', 'FEMALE ', 'Male', 'Female', 'Male', 'Female', 'MALE', 'Female', 'Male'],
    'MonthlyBill': [45.5, 30.2, np.nan, 70.1, 55.0, 150.0, 20.0, 62.5, 48.0, 250.0, 50.0, 35.0, 1000.0, 42.0, 68.0],
    'DataUsageGB': [10, 5, 25, 15, 8, 30, np.nan, 20, 12, 50, 18, 7, 40, 11, np.nan],
    'PlanType': ['Basic', 'Premium', 'basic', 'Premium', 'BASIC', 'Premium', 'Basic', 'Premium', 'Basic', 'Ultimate', 'Premium', 'Basic', 'ultimate', 'Basic', 'Premium'],
    'Churn': ['No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes']
}
df_exercise = pd.DataFrame(exercise_data)

print("--- Exercise DataFrame Initial State ---")
df_exercise.info()
print("\nMissing values:\n", df_exercise.isnull().sum())
for col in ['Age', 'Gender', 'MonthlyBill', 'DataUsageGB', 'PlanType']:
    if df_exercise[col].dtype == 'object' or df_exercise[col].nunique() < 10: # Check for unique values in categorical
        print(f"Unique values for {col}: {df_exercise[col].unique()}")
print("-" * 30)

"""
1. Identify All Issues:

    Using df_exercise.info(), isnull().sum(), unique(), and describe(), identify all missing values, potential outliers, and inconsistencies in data types, capitalization, and logical ranges within the df_exercise DataFrame. List them out.
"""

print("--- Exercise DataFrame Initial State ---")
df_exercise.info()
print("\nMissing values:\n", df_exercise.isnull().sum())
for col in ['Age', 'Gender', 'MonthlyBill', 'DataUsageGB', 'PlanType']:
    if df_exercise[col].dtype == 'object' or df_exercise[col].nunique() < 10: # Check for unique values in categorical
        print(f"Unique values for {col}: {df_exercise[col].unique()}")
print("-" * 30)

# Identified Issues:
# - Missing values: Age (1), MonthlyBill (1), DataUsageGB (2)
# - Data types: Age is float64 due to NaN, but should be int after cleaning; others seem appropriate.
# - Inconsistencies in capitalization: Gender has 'male', 'Female', 'MALE', 'FEMALE ' (with space), 'Male'; PlanType has 'Basic', 'Premium', 'basic', 'BASIC', 'Ultimate', 'ultimate'.
# - Logical ranges: Age has negative (-5) and extremely high (150) values; MonthlyBill has potential outliers (e.g., 1000.0); DataUsageGB seems reasonable.
# - Outliers: MonthlyBill likely has outliers based on describe (max 1000 vs mean ~100); Age 150 is an outlier (human lifespan max ~120, so treat as error).


"""
2. Handle Missing Values:

    For Age and DataUsageGB, impute missing values using the median.
    For MonthlyBill, observe its unique values and distribution (e.g., with a histogram). Decide if mean or median imputation is better, and justify your choice. Then apply the chosen imputation.
"""

print("\n--- Step 2: Handle Missing Values ---")

# Impute Age and DataUsageGB with median
age_median = df_exercise['Age'].median()
df_exercise['Age'] = df_exercise['Age'].fillna(age_median)
print(f"Imputed Age missing value with median: {age_median}")

data_usage_median = df_exercise['DataUsageGB'].median()
df_exercise['DataUsageGB'] = df_exercise['DataUsageGB'].fillna(data_usage_median)
print(f"Imputed DataUsageGB missing value with median: {data_usage_median}")

# For MonthlyBill: Observe unique values and distribution
print("MonthlyBill unique values:", df_exercise['MonthlyBill'].unique())
print("MonthlyBill describe:\n", df_exercise['MonthlyBill'].describe())

# Plot histogram for MonthlyBill distribution
# plt.figure(figsize=(10, 6))
# df_exercise['MonthlyBill'].hist(bins=30, edgecolor='black')
# plt.title('MonthlyBill Distribution')
# plt.xlabel('MonthlyBill')
# plt.ylabel('Frequency')
# plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df_exercise['MonthlyBill'].dropna(), bins=10, edgecolor='black')
plt.title('MonthlyBill Distribution')
plt.xlabel('MonthlyBill')
plt.ylabel('Frequency')
plt.show()


# Decision: The distribution shows a right-skew (mean > median), with potential outliers (e.g., 1000). Median is more robust to outliers, so use median imputation.
monthly_bill_median = df_exercise['MonthlyBill'].median()
df_exercise['MonthlyBill'] = df_exercise['MonthlyBill'].fillna(monthly_bill_median)
print(f"Imputed MonthlyBill missing value with median: {monthly_bill_median} (chosen because distribution is skewed, median is robust to outliers)")

"""
3. Correct Inconsistencies:

    Standardize the Gender column (e.g., to "Male", "Female").
    Standardize the PlanType column (e.g., to "Basic", "Premium", "Ultimate").
    Correct the logical error in the Age column where Age is negative by replacing it with a reasonable value (e.g., the median age, or 0 if it represents unknown/error). Also, identify and consider the extremely high Age (150) - is it an outlier or a logical error? Decide how to handle it.
"""
print("\n--- Step 3: Correct Inconsistencies ---")

# Standardize Gender
df_exercise['Gender'] = df_exercise['Gender'].str.strip().str.capitalize()
print("Gender standardized to:", df_exercise['Gender'].unique())

# Standardize PlanType
df_exercise['PlanType'] = df_exercise['PlanType'].str.strip().str.capitalize()
print("PlanType standardized to:", df_exercise['PlanType'].unique())

# Correct logical error in Age
age_median_clean = df_exercise['Age'].median()
df_exercise['Age'] = df_exercise['Age'].apply(lambda x: age_median_clean if x < 0 else x)
print(f"Replaced negative Age with median: {age_median_clean}")

# Handle extremely high Age (150)
df_exercise['Age'] = df_exercise['Age'].apply(lambda x: age_median_clean if x > 120 else x)
print(f"Replaced Age > 120 (150) with median: {age_median_clean} (assuming it's an error)")

df_exercise['Age'] = df_exercise['Age'].astype(int)

"""
4. Detect and Treat Outliers:

    For the MonthlyBill column, use a box plot to visualize outliers.
    Apply the IQR method to identify numerical outliers in MonthlyBill.
    Cap these outliers within the IQR bounds (lower bound and upper bound).
"""
print("\n--- Step 4: Detect and Treat Outliers ---")
# Box plot for MonthlyBill
plt.figure(figsize=(8, 5))
plt.boxplot(df_exercise['MonthlyBill'], vert=False)
plt.title('MonthlyBill Box Plot')
plt.xlabel('MonthlyBill')
plt.show()

# IQR method for MonthlyBill
Q1 = df_exercise['MonthlyBill'].quantile(0.25)
Q3 = df_exercise['MonthlyBill'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"MonthlyBill IQR: Q1={Q1}, Q3={Q3}, IQR={IQR}, Lower={lower_bound}, Upper={upper_bound}")
# Identify outliers
outliers = df_exercise[(df_exercise['MonthlyBill'] < lower_bound) | (df_exercise['MonthlyBill'] > upper_bound)]
print("Outliers in MonthlyBill:\n", outliers[['CustID', 'MonthlyBill']])
# Cap outliers within bounds
df_exercise['MonthlyBill'] = np.where(df_exercise['MonthlyBill'] < lower_bound, lower_bound,
                                     np.where(df_exercise['MonthlyBill'] > upper_bound, upper_bound, df_exercise['MonthlyBill']))
print("Capped MonthlyBill outliers.")

"""
5. Verify Cleaning:

    After performing all cleaning steps, run df_exercise.info(), isnull().sum(), unique(), and describe() again. Confirm that all identified issues have been resolved.
""" 

print("\n--- Step 5: Verify Cleaning ---")
print("DataFrame Info after cleaning:")
df_exercise.info()
print("\nMissing values after cleaning:\n", df_exercise.isnull().sum())
print("\nDescribe numerical columns after cleaning:")
print(df_exercise.describe())
for col in ['Age', 'Gender', 'MonthlyBill', 'DataUsageGB', 'PlanType', 'Churn']:
    if df_exercise[col].dtype == 'object' or df_exercise[col].nunique() < 10:
        print(f"Unique values for {col}: {df_exercise[col].unique()}")