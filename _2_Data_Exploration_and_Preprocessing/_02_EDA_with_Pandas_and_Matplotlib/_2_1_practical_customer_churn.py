import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Often useful for numerical operations, though not central to EDA visuals here

# Load the dataset
try:
    df = pd.read_csv('customer_churn.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'customer_churn.csv' not found. Please ensure the file is in the correct directory.")
    # Create a dummy DataFrame for demonstration if file not found
    data = {
        'CustomerID': [f'C{i:04d}' for i in range(1, 21)],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
                   'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'SeniorCitizen': [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        'Partner': ['Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No'],
        'Dependents': ['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'],
        'Tenure': [1, 34, 2, 45, 2, 8, 22, 10, 28, 62, 13, 16, 58, 49, 10, 1, 69, 70, 21, 1],
        'PhoneService': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No'],
        'MultipleLines': ['No phone service', 'No', 'No', 'No phone service', 'No', 'Yes', 'Yes', 'No phone service', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No phone service', 'No', 'No', 'Yes', 'No', 'No phone service'],
        'InternetService': ['DSL', 'DSL', 'DSL', 'DSL', 'Fiber optic', 'Fiber optic', 'Fiber optic', 'DSL', 'Fiber optic', 'DSL', 'DSL', 'Fiber optic', 'Fiber optic', 'Fiber optic', 'DSL', 'DSL', 'Fiber optic', 'Fiber optic', 'Fiber optic', 'DSL'],
        'OnlineSecurity': ['No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No'],
        'OnlineBackup': ['Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
        'DeviceProtection': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No'],
        'TechSupport': ['No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No'],
        'StreamingTV': ['No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
        'StreamingMovies': ['No', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'One year', 'Month-to-month', 'Month-to-month', 'Month-to-month', 'Month-to-month', 'Month-to-month', 'One year', 'Month-to-month', 'Two year', 'One year', 'Month-to-month', 'Month-to-month', 'Month-to-month', 'Two year', 'Two year', 'Month-to-month', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check', 'Bank transfer (automatic)', 'Electronic check', 'Electronic check', 'Credit card (automatic)', 'Mailed check', 'Electronic check', 'Bank transfer (automatic)', 'Mailed check', 'Credit card (automatic)', 'Bank transfer (automatic)', 'Electronic check', 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Electronic check'],
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 45.90, 90.25, 50.55, 76.50, 78.95, 110.50, 84.80, 49.30, 20.20, 103.70, 109.20, 90.05, 29.85],
        'TotalCharges': ['29.85', '1889.5', '108.15', '1840.75', '8.6', '684.4', '1949.4', '301.9', '2807.6', '3487.95', '587.45', '1146.8', '333.7', '4759.55', '929.85', '20.2', '7639.45', '7709.65', '1862.9', '29.85'],
        'Churn': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No', 'No', 'Yes']
    }
    
    df = pd.DataFrame(data)
    print("Using dummy dataset for demonstration.")

# Display the first few rows to get a quick overview
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display information about the dataset
print("\nDataFrame Info:")
df.info()

# Convert TotalCharges to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Re-check info after conversion
print("\nDataFrame Info after converting 'TotalCharges':")
df.info()

# Display descriptive statistics for numerical columns
print("\nDescriptive statistics for numerical columns:")
df.describe()

# Display descriptive statistics for categorical columns
print("\nDescriptive statistics for categorical columns:")
df.describe(include='object')

# Unique values and counts
# Check unique values and their counts for 'Gender'
print("\nValue counts for 'Gender':")
print(df['Gender'].value_counts())

# Check unique values and their counts for 'Contract'
print("\nValue counts for 'Contract':")
print(df['Contract'].value_counts())

# Check number of unique values in 'PaymentMethod'
print("\nNumber of unique payment methods:", df['PaymentMethod'].nunique())

# Visualizing Data Distributions and Relationships with Matplotlib
print("\n\nVisualizing Data Distributions and Relationships with Matplotlib:")

# Univariate Analysis (Single Variable)

# Histogram for MonthlyCharges
plt.figure(figsize=(8, 5))
plt.hist(df['MonthlyCharges'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Number of Customers')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histogram for Tenure
plt.figure(figsize=(8, 5))
plt.hist(df['Tenure'], bins=20, color='lightcoral', edgecolor='black')
plt.title('Distribution of Customer Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Box Plots for Numerical Data

# Box plot for MonthlyCharges
plt.figure(figsize=(8, 5))
plt.boxplot(df['MonthlyCharges'])
plt.title('Box Plot of Monthly Charges')
plt.ylabel('Monthly Charges')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Box plot for TotalCharges (after handling NaNs for visualization)
# For simplicity, we drop NaNs for this specific visualization.
# In data cleaning, you'd handle them more robustly.
plt.figure(figsize=(8, 5))
plt.boxplot(df['TotalCharges'].dropna()) # dropna() to handle NaN values for plotting
plt.title('Box Plot of Total Charges')
plt.ylabel('Total Charges')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Bar Charts for Categorical Data

# Bar chart for Gender distribution
plt.figure(figsize=(6, 4))
df['Gender'].value_counts().plot(kind='bar', color=['lightseagreen', 'palevioletred'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0) # Keep labels horizontal
plt.grid(axis='y', alpha=0.75)
plt.show()

# Bar chart for Internet Service distribution
plt.figure(figsize=(8, 5))
df['InternetService'].value_counts().plot(kind='bar', color='darkorange')
plt.title('Distribution of Internet Service Types')
plt.xlabel('Internet Service Type')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45) # Rotate labels for readability
plt.grid(axis='y', alpha=0.75)
plt.show()

# Bivariate Analysis (Two Variables)
print("\n\nBivariate Analysis (Two Variables):")

# Scatter plot between MonthlyCharges and TotalCharges
plt.figure(figsize=(10, 6))
plt.scatter(df['MonthlyCharges'], df['TotalCharges'], alpha=0.6, color='darkblue')
plt.title('Monthly Charges vs. Total Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Comparing Numerical by Categorical Variables

# Box plot of MonthlyCharges by Churn status
plt.figure(figsize=(8, 6))
df.boxplot(column='MonthlyCharges', by='Churn', grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Monthly Charges by Churn Status')
plt.suptitle('') # Suppress the default matplotlib title for 'by' parameter
plt.xlabel('Churn Status')
plt.ylabel('Monthly Charges')
plt.show()

# Box plot of Tenure by InternetService
plt.figure(figsize=(10, 6))
df.boxplot(column='Tenure', by='InternetService', grid=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Customer Tenure by Internet Service Type')
plt.suptitle('')
plt.xlabel('Internet Service Type')
plt.ylabel('Tenure (Months)')
plt.show()

# Correlation Analysis for Numerical Variables

# Calculate the correlation matrix for numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numerical_cols].corr()

print("\nCorrelation Matrix for Numerical Features:")
print(correlation_matrix)

# Visualize the correlation matrix (basic Matplotlib approach)
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Grouped Analysis with Pandas
print("\n\nGrouped Analysis with Pandas:")

# Calculate the average MonthlyCharges for each Gender
print("\nAverage Monthly Charges by Gender:")
print(df.groupby('Gender')['MonthlyCharges'].mean())

# Calculate the average Tenure for each Contract type
print("\nAverage Tenure by Contract Type:")
print(df.groupby('Contract')['Tenure'].mean())

# Churn rates by Internet Service (requires converting 'Churn' to numeric first)
# For this example, let's map 'Yes' to 1 and 'No' to 0
df['Churn_numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

print("\nChurn Rate by Internet Service:")
print(df.groupby('InternetService')['Churn_numeric'].mean())

# Visualize churn rate by Internet Service
churn_rate_by_internet = df.groupby('InternetService')['Churn_numeric'].mean()
plt.figure(figsize=(8, 5))
churn_rate_by_internet.plot(kind='bar', color='purple')
plt.title('Churn Rate by Internet Service')
plt.xlabel('Internet Service Type')
plt.ylabel('Churn Rate (Proportion)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.show()