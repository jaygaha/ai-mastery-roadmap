"""
3. Loading Specific Columns:

    Load _2_3_customer_churn.csv again, but this time, only load the CustomerID, Gender, and MonthlyCharges columns.
    Set CustomerID as the index for this DataFrame.
    Print the head and dtypes of the new DataFrame.
"""

import pandas as pd
import io

# 1. Recreate the dummy CSV file content locally for execution
# Assuming a standard customer churn CSV structure (same as previous examples for consistency)
csv_content = """CustomerID,Gender,Age,MonthlyCharges,Churn_Status
C101,Male,45,65.75,No
C102,Female,32,90.20,Yes
C103,Male,60,45.00,No
C104,Female,28,105.50,Yes
C105,Male,39,78.90,No
C106,Female,51,19.95,No
"""

file_path = '_2_3_customer_churn.csv'
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(csv_content)

print(f"Created file: {file_path}\n")

# --- Load only specific columns and set CustomerID as index ---

# Use the usecols parameter to specify which columns to load.
# Use the index_col parameter to specify which column should be used as the DataFrame index.
df_subset = pd.read_csv(
    file_path,
    usecols=['CustomerID', 'Gender', 'MonthlyCharges'],
    index_col='CustomerID'
)

# --- Print the head and dtypes of the new DataFrame ---

print("First 5 rows of the subset DataFrame:\n", df_subset.head())

print("\nData Types (dtypes) of the subset DataFrame:\n", df_subset.dtypes)

print("\nIndex of the subset DataFrame:\n", df_subset.index)


# Clean up the created file (optional)
import os
os.remove(file_path)
print(f"\nCleaned up file: {file_path}")
