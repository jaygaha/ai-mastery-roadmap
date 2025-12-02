import pandas as pd

# Create a dummy Excel file (requires openpyxl or xlrd installed: pip install openpyxl)
from pandas import ExcelWriter

excel_data = {
    'Sheet1_Churn_Data': pd.DataFrame({
        'CustomerID': [1001, 1002, 1003],
        'MonthlyCharges': [50.00, 80.50, 30.00],
        'Churn': ['No', 'Yes', 'No']
    }),
    'Sheet2_Churn_Details': pd.DataFrame({
        'CustomerID': [1001, 1002, 1003],
        'Contract': ['Month-to-month', 'Two year', 'Month-to-month']
    })
}

with ExcelWriter('_1_3_customer_churn.xlsx') as writer:
    excel_data['Sheet1_Churn_Data'].to_excel(writer, sheet_name='Sheet1_Churn_Data', index=False)
    excel_data['Sheet2_Churn_Details'].to_excel(writer, sheet_name='Sheet2_Churn_Details', index=False)

# Load a specific sheet from the Excel file
df_excel_churn = pd.read_excel('_1_3_customer_churn.xlsx', sheet_name='Sheet1_Churn_Data')
print("\nDataFrame loaded from Excel (Sheet1_Churn_Data):")
print(df_excel_churn.head())

# Load another sheet
df_excel_details = pd.read_excel('_1_3_customer_churn.xlsx', sheet_name='Sheet2_Churn_Details')
print("\nDataFrame loaded from Excel (Sheet2_Churn_Details):")
print(df_excel_details.head())