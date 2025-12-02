import pandas as pd

# Create a dummy CSV file for demonstration
csv_data = """CustomerID,Gender,Age,MonthlyCharges,TotalCharges,Churn
1001,Male,34,50.00,1700.00,No
1002,Female,56,80.50,4500.25,Yes
1003,Female,22,30.00,30.00,No
1004,Male,45,100.25,6000.50,Yes
1005,Male,67,25.75,100.00,No
"""

with open('_1_1_customer_churn.csv', 'w') as f:
    f.write(csv_data)

# Load the CSV file into a Pandas DataFrame
df_churn = pd.read_csv('_1_1_customer_churn.csv')

# Display the first few rows
print("DataFrame loaded from CSV:")
print(df_churn.head())
print("\nData types of columns:")
print(df_churn.dtypes)