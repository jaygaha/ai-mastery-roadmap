import pandas as pd

# Example 2: DataFrame from a list of dictionaries
# Each dictionary in the list becomes a row. Keys become column names.
employee_data = [
    { 'Name': 'Hoge', 'Age': 30, 'Department': 'HR', 'Salary': 50000},
    { 'Name': 'Fuga', 'Age': 25, 'Department': 'IT', 'Salary': 60000},
    { 'Name': 'Moge', 'Age': 35, 'Department': 'Marketing', 'Salary': 55000}
]

employees_df = pd.DataFrame(employee_data)
print("\nDataFrame from a list of dictionaries:")
print(employees_df)