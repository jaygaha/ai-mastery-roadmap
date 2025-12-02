"""
2. Customer Churn Case Study - Data Loading and Inspection:

Using the _2_2_customer_churn_complex.csv file created in the "Loading from CSV" section:

    Load the file into a DataFrame.
    Explicitly set the Churn_Status column to the bool data type. You might need to map 'Yes' to True and 'No' to False. (Hint: map() or conditional assignment can be useful after initial loading, or explore the true_values and false_values parameters of pd.read_csv).
    After loading, display the first 5 rows and verify the dtypes of all columns.
    Can you calculate the average MonthlyCharges for customers who churned (Churn_Status is True)? (This hints at future EDA, but tests correct data type loading).

"""

import pandas as pd
import io
import numpy as np

# 1. Recreate the dummy CSV file content locally for execution
# This content matches the file described in the "Loading from CSV" prompt of the case study.
csv_content = """CustomerID,Gender,Age,MonthlyCharges,Churn_Status
C101,Male,45,65.75,No
C102,Female,32,90.20,Yes
C103,Male,60,45.00,No
C104,Female,28,105.50,Yes
C105,Male,39,78.90,No
C106,Female,51,19.95,No
"""

file_path = '_2_2_customer_churn_complex.csv'
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(csv_content)

print(f"Created file: {file_path}\n")

# --- Load the file with specific data type handling ---

# Use the true_values and false_values parameters in pd.read_csv to map 'Yes'/'No' directly to True/False booleans during load.
df = pd.read_csv(
    file_path,
    true_values=['Yes'],
    false_values=['No']
)

# --- Display the first 5 rows and verify dtypes ---

print("First 5 rows of the DataFrame:\n", df.head())

print("\nData Types (dtypes) of all columns:\n", df.dtypes)

# The Churn_Status column is now correctly loaded as boolean (bool) type.


# --- Calculate the average MonthlyCharges for customers who churned ---

# We can now use boolean indexing directly because Churn_Status is already a boolean column.

churned_customers = df[df['Churn_Status'] == True]
average_churn_charges = churned_customers['MonthlyCharges'].mean()

# Alternatively, using a direct one-liner:
average_churn_charges_concise = df[df['Churn_Status']]['MonthlyCharges'].mean()

print(f"\nAverage MonthlyCharges for customers who churned (Churn_Status is True): ${average_churn_charges:.2f}")


# Clean up the created file (optional)
import os
os.remove(file_path)
print(f"\nCleaned up file: {file_path}")