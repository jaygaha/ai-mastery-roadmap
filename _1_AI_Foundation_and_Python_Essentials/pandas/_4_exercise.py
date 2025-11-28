"""
Exercises and Practice Activities

To solidify your understanding of Pandas, try these exercises using the df_purchases DataFrame (the one after adding TotalAmount and introducing 
the NaN for 'Keyboard' price, but before dropna() or fillna() operations for the exercises that follow).

"""

import pandas as pd
import numpy as np
import io

csv_data_initial = """CustomerID,ProductName,Category,Price,Quantity,PurchaseDate
1001,Laptop A,Electronics,1200.00,1,2025-11-15
1002,Smartphone X,Electronics,800.00,2,2025-11-16
1001,Mouse,Accessories,25.50,1,2025-11-17
1003,Keyboard,Accessories,75.00,1,2025-11-18
1002,Headphones,Electronics,150.00,1,2025-11-19
1004,Webcam,Accessories,50.00,1,2025-11-20
1003,Monitor,Electronics,300.00,1,2025-11-21
1001,SSD Drive,Components,80.00,2,2025-11-22
"""
df_purchases_original = pd.read_csv(io.StringIO(csv_data_initial))

# Add TotalAmount and introduce NaN as in the demonstration for exercises
df_purchases = df_purchases_original.copy()  # Create a copy to avoid modifying original
df_purchases['TotalAmount'] = df_purchases['Price'] * df_purchases['Quantity']
df_purchases.loc[3, 'Price'] = np.nan  # Make price missing for the 'Keyboard' purchase

"""
Exercise 1: 

Inspect and Filter:

    Display the .info() and .describe() output for the current df_purchases.
    Find all purchases made by CustomerID 1002.
    Filter the DataFrame to show only items with Price greater than $100.
    Show all purchases that are either in the 'Accessories' category or have a Quantity of 2.
"""

# Display the .info() and .describe() output for the current df_purchases.

print("DataFrame Info:")
print(df_purchases.info())
print("\nDataFrame Describe:")
print(df_purchases.describe())

"""
Explanation: .info() shows the structure, including data types and non-null counts (revealing the NaN in Price and TotalAmount). 
.describe() provides summary statistics for numeric columns, excluding NaN values by default.
"""

# Find all purchases made by CustomerID 1002.

purchases_1002 = df_purchases[df_purchases['CustomerID'] == 1002]
print(purchases_1002)


"""
Explanation: This uses boolean indexing to filter rows where CustomerID equals 1002.
"""

# Filter the DataFrame to show only items with Price greater than $100.
expensive_items = df_purchases[df_purchases['Price'] > 100]
print(expensive_items)

"""
Explanation: Again, boolean indexing filters rows where Price > 100. Note that the row with NaN in Price (Keyboard) is excluded since NaN comparisons are False.
"""

# Show all purchases that are either in the 'Accessories' category or have a Quantity of 2.
filtered_purchases = df_purchases[(df_purchases['Category'] == 'Accessories') | (df_purchases['Quantity'] == 2)]
print(filtered_purchases)

"""
Explanation: Uses the | (OR) operator with boolean conditions. This includes Accessories and any row with Quantity=2.
"""

"""
Exercise 2:

Column Operations:
    Add a new column called IsExpensive that is True if Price is greater than $200 and False otherwise.
    Create a new column called Month which extracts the month number from the PurchaseDate. (Hint: You might need to convert PurchaseDate to datetime objects first, using pd.to_datetime(), then access the .dt.month attribute).
    Delete the Category column temporarily and confirm it's gone (without modifying the original df_purchases in place).
"""

# Add a new column called IsExpensive that is True if Price is greater than $200 and False otherwise.
df_purchases['IsExpensive'] = df_purchases['Price'] > 200
print(df_purchases[['ProductName', 'Price', 'IsExpensive']])  # Show relevant columns

"""
Explanation: Creates a boolean column based on the condition. NaN comparisons result in False.
"""

# Create a new column called Month which extracts the month number from the PurchaseDate. (Hint: You might need to convert PurchaseDate to datetime objects first, 
# using pd.to_datetime(), then access the .dt.month attribute).

df_purchases['PurchaseDate'] = pd.to_datetime(df_purchases['PurchaseDate'])
df_purchases['Month'] = df_purchases['PurchaseDate'].dt.month
print(df_purchases[['ProductName', 'PurchaseDate', 'Month']])  # Show relevant columns

"""
Explanation: Converts PurchaseDate to datetime, then extracts the month. All dates are in January (month 1).
"""

# Delete the Category column temporarily and confirm it's gone (without modifying the original df_purchases in place).

df_temp = df_purchases.drop('Category', axis=1)  # axis=1 for columns
print("Columns after dropping Category:")
print(df_temp.columns.tolist())
print("\nOriginal df_purchases still has Category:")
print('Category' in df_purchases.columns)  # Should be True

"""
Explanation: .drop() creates a new DataFrame without the column unless inplace=True. The original remains unchanged.
"""

"""
Exercise 3:

Grouping and Aggregation:
    Group purchases by CustomerID and calculate the total amount spent by each customer.
    Find the average price of purchases in each category.
    Calculate the total quantity of items purchased for each month.
"""

# Group purchases by CustomerID and calculate the total amount spent by each customer.
grouped_customers = df_purchases.groupby('CustomerID')['TotalAmount'].sum()
print(grouped_customers)

"""
Explanation: Groups by CustomerID and uses .sum() to calculate total amounts for each group.
"""

# Find the average price of purchases in each category.
grouped_categories = df_purchases.groupby('Category')['Price'].mean()
print(grouped_categories)

"""
Explanation: Groups by Category and uses .mean() to calculate average prices for each category.
"""

# Calculate the total quantity of items purchased for each month.
grouped_months = df_purchases.groupby('Month')['Quantity'].sum()
print(grouped_months)

"""
Explanation: Groups by Month and uses .sum() to calculate total quantities for each month.
"""

"""
Exercise 4:

Missing Values and Aggregation


    Count the total number of missing values in the entire df_purchases DataFrame.
    Fill any missing values in the Price column with the median price of all products.
    Group the DataFrame by Category and find the average Quantity purchased for each category.
    Determine which CustomerID has the highest total TotalAmount spent.

"""

# Count the total number of missing values in the entire df_purchases DataFrame.
total_missing = df_purchases.isnull().sum().sum()
print(f"Total missing values: {total_missing}")

"""
Explanation: .isnull().sum().sum() counts NaNs across all cells. There are 2: one in Price and one in TotalAmount (since TotalAmount depends on Price).
"""

# Fill any missing values in the Price column with the median price of all products.
median_price = df_purchases['Price'].median()
df_purchases['Price'].fillna(median_price, inplace=True)
print(f"Median Price used: {median_price}")
print(df_purchases[['ProductName', 'Price']])  # Show updated Price column

"""
Explanation: Calculates the median (80.0) and fills NaN in Price. Note: This modifies the DataFrame in place.
"""

# Group the DataFrame by Category and find the average Quantity purchased for each category.
avg_quantity_by_category = df_purchases.groupby('Category')['Quantity'].mean()
print(avg_quantity_by_category)

"""
Explanation: Groups by Category and computes the mean of Quantity. 
Accessories average is 1.0 (3 items, all qty=1), 
Components is 2.0 (1 item), 
Electronics is ~1.33 (3 items: 1,2,1).
"""

# Determine which CustomerID has the highest total TotalAmount spent.
# First, ensure TotalAmount is updated if Price was filled (from previous step)
df_purchases['TotalAmount'] = df_purchases['Price'] * df_purchases['Quantity']
total_spent = df_purchases.groupby('CustomerID')['TotalAmount'].sum()
top_customer = total_spent.idxmax()
top_amount = total_spent.max()
print(f"CustomerID with highest total spent: {top_customer} (${top_amount})")

"""
Explanation: Groups by CustomerID, sums TotalAmount, and finds the max. Customer 1002 spent the most (1600 + 150 = 1750, now that Keyboard's Price is filled, but it didn't affect 1002).
"""