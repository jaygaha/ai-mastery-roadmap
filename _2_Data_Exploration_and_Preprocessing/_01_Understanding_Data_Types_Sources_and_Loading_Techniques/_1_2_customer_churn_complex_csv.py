import pandas as pd

# Create a more complex dummy CSV file to demonstrate parameter usage
csv_data_complex = """CustomerID,Gender,Age,MonthlyCharges,TotalCharges,Churn_Status,JoinDate
1001,Male,34,50.00,1700.00,No,2024-01-15
1002,Female,56,80.50,4500.25,Yes,2020-03-22
1003,Female,22,30.00,,No,2023-07-01
1004,Male,45,100.25,6000.50,Yes,2021-11-05
1005,Male,67,25.75,100.00,No,2022-09-30
"""

with open('_1_2_customer_churn_complex.csv', 'w') as f:
    f.write(csv_data_complex)

# Load the complex CSV file, specifying dtypes, na_values, and parsing dates
df_churn_complex = pd.read_csv(
    '_1_2_customer_churn_complex.csv',
    sep=',',
    header=0,
    index_col='CustomerID', # Use CustomerID as the DataFrame index
    dtype={'Gender': 'category', 'Age': 'int64', 'MonthlyCharges': 'float64'}, # Explicitly set dtypes
    na_values=[' '], # Treat empty strings as NaN for numeric conversion
    parse_dates=['JoinDate'] # Parse 'JoinDate' column as datetime objects
)

print("\nDataFrame loaded from complex CSV with parameters:")
print(df_churn_complex.head())
print("\nData types of columns after advanced loading:")
print(df_churn_complex.dtypes)

# Notice 'TotalCharges' is now float64, even with a missing value, because we specified na_values.
# 'Churn_Status' is still 'object' and can be converted later to 'bool' or 'category'.