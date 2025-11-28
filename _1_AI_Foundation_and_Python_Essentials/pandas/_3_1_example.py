import pandas as pd
import io # Used to simulate a file from a string
import numpy as np

# Simulate a CSV file content
csv_data = """CustomerID,ProductName,Category,Price,Quantity,PurchaseDate
1001,Laptop A,Electronics,1200.00,1,2025-11-15
1002,Smartphone X,Electronics,800.00,2,2025-11-16
1001,Mouse,Accessories,25.50,1,2025-11-17
1003,Keyboard,Accessories,75.00,1,2025-11-18
1002,Headphones,Electronics,150.00,1,2025-11-19
1004,Webcam,Accessories,50.00,1,2025-11-20
1003,Monitor,Electronics,300.00,1,2025-11-21
1001,SSD Drive,Components,80.00,2,2025-11-22
"""

# Use io.StringIO to treat the string as a file
df_purchases = pd.read_csv(io.StringIO(csv_data))

print("DataFrame loaded from CSV:\n")
print(df_purchases)
print("\nType of df_purchases:", type(df_purchases))

# Display the first 5 rows (default)
print("\nFirst 3 rows of the DataFrame:")
print(df_purchases.head(3)) # You can specify the number of rows

# Display the last 2 rows
print("\nLast 2 rows of the DataFrame:")
print(df_purchases.tail(2))

print("\nDataFrame Info:")
df_purchases.info()

print("\nShape of the DataFrame (rows, columns):", df_purchases.shape)
print("Column names:", df_purchases.columns)

print("\nDescriptive statistics for numerical columns:")
print(df_purchases.describe())

print("\n\nSelecting Data (Indexing and Slicing)\n\n")

# Select 'ProductName' column using dictionary-like syntax
product_names = df_purchases['ProductName']
print("\n'ProductName' column (Series):\n", product_names.head())
print("Type of product_names:", type(product_names))

# Alternatively, using dot notation (if column name is a valid Python identifier)
# This is concise but can be ambiguous if column name clashes with DataFrame methods.
prices = df_purchases.Price
print("\n'Price' column (Series) using dot notation:\n", prices.head())

# Select 'ProductName' and 'Price' columns
products_and_prices = df_purchases[['ProductName', 'Price']]
print("\n'ProductName' and 'Price' columns (DataFrame):\n", products_and_prices.head())
print("Type of products_and_prices:", type(products_and_prices))

# Select the first row (index 0)
first_row = df_purchases.iloc[0]
print("\nFirst row using .iloc[0]:\n", first_row)
print("Type of first_row:", type(first_row)) # A Series

# Select rows from index 1 up to (but not including) 4
rows_1_to_3 = df_purchases.iloc[1:4]
print("\nRows from index 1 to 3 using .iloc[1:4]:\n", rows_1_to_3)

# Select specific non-contiguous rows and columns (e.g., rows 0, 2, and columns 0, 2)
specific_selection = df_purchases.iloc[[0, 2], [0, 2]]
print("\nSpecific rows (0, 2) and columns (0, 2) using .iloc[[0, 2], [0, 2]]:\n", specific_selection)

# Select row with index label 0 (same as iloc[0] for default integer index)
row_by_label_0 = df_purchases.loc[0]
print("\nRow with index label 0 using .loc[0]:\n", row_by_label_0)

# Select rows from label 1 to label 3 (inclusive with .loc)
# This is a key difference: .loc slicing is inclusive of the stop label
rows_label_1_to_3 = df_purchases.loc[1:3]
print("\nRows from label 1 to 3 (inclusive) using .loc[1:3]:\n", rows_label_1_to_3)

# Select specific rows and columns by labels
# e.g., rows with label 0 and 2, columns 'ProductName' and 'Price'
specific_label_selection = df_purchases.loc[[0, 2], ['ProductName', 'Price']]
print("\nSpecific rows (0, 2) and columns ('ProductName', 'Price') using .loc:\n", specific_label_selection)

print("\n\nFiltering Data (Boolean Indexing)\n\n")

# Select all purchases where the 'Quantity' is greater than 1
high_quantity_purchases = df_purchases[df_purchases['Quantity'] > 1]
print("\nPurchases with Quantity > 1:\n", high_quantity_purchases)

# Select all purchases in the 'Electronics' category
electronics_purchases = df_purchases[df_purchases['Category'] == 'Electronics']
print("\nElectronics purchases:\n", electronics_purchases)

# Combine multiple conditions using logical operators (& for AND, | for OR)
# Purchases in 'Electronics' category AND Price > 500
expensive_electronics = df_purchases[
    (df_purchases['Category'] == 'Electronics') &
    (df_purchases['Price'] > 500)
]
print("\nExpensive Electronics purchases:\n", expensive_electronics)

# Purchases by CustomerID 1001 OR in 'Accessories' category
customer_1001_or_accessories = df_purchases[
    (df_purchases['CustomerID'] == 1001) |
    (df_purchases['Category'] == 'Accessories')
]
print("\nPurchases by Customer 1001 OR in Accessories:\n", customer_1001_or_accessories)

# Using .isin() for multiple discrete values
# Select products that are either 'Laptop A' or 'Monitor'
specific_products = df_purchases[df_purchases['ProductName'].isin(['Laptop A', 'Monitor'])]
print("\nSpecific products ('Laptop A', 'Monitor'):\n", specific_products)

print("\n\nAdding, Modifying, and Deleting Columns\n\n")

# Add a 'TotalAmount' column (Price * Quantity)
df_purchases['TotalAmount'] = df_purchases['Price'] * df_purchases['Quantity']
print("\nDataFrame with 'TotalAmount' column added:\n", df_purchases)

# Add a 'DiscountedPrice' column (e.g., 10% discount)
df_purchases['DiscountedPrice'] = df_purchases['Price'] * 0.90
print("\nDataFrame with 'DiscountedPrice' column added:\n", df_purchases)

# Increase the price of all accessories by 5%
# We first select the rows where Category is 'Accessories'
# Then we select the 'Price' column for those rows and update it.
df_purchases.loc[df_purchases['Category'] == 'Accessories', 'Price'] *= 1.05
print("\nDataFrame after increasing 'Accessories' prices by 5%:\n", df_purchases)

print("\n\nHandling Missing Values (Basic Introduction)")


# Introduce a NaN value for demonstration
df_purchases.loc[3, 'Price'] = np.nan # Make price missing for the 'Keyboard' purchase

print("\nDataFrame with a missing 'Price' value:\n", df_purchases)

# Check for missing values using .isnull()
print("\nDataFrame indicating missing values (.isnull()):\n", df_purchases.isnull())

# Count missing values per column
print("\nNumber of missing values per column:\n", df_purchases.isnull().sum())

# Drop rows with any missing values
# Caution: This can remove a lot of data if many rows have missing values.
df_no_nan_rows = df_purchases.dropna()
print("\nDataFrame after dropping rows with any NaN values:\n", df_no_nan_rows)

# Fill missing values with a specific value (e.g., 0)
df_filled_zero = df_purchases.fillna(0)
print("\nDataFrame after filling NaN with 0:\n", df_filled_zero)

# Fill missing numerical values with the mean of the column
# We only fill the 'Price' column's NaN values
mean_price = df_purchases['Price'].mean()
df_filled_mean = df_purchases.fillna({'Price': mean_price})
print(f"\nDataFrame after filling 'Price' NaN with mean ({mean_price:.2f}):\n", df_filled_mean)


print("\n\nBasic Aggregations\n\n")

# Calculate the total sales (sum of 'TotalAmount')
total_sales = df_purchases['TotalAmount'].sum()
print(f"\nTotal sales across all purchases: ${total_sales:.2f}")

# Calculate the average price of all products
average_price = df_purchases['Price'].mean()
print(f"Average product price: ${average_price:.2f}")

# Calculate the maximum quantity purchased in a single transaction
max_quantity = df_purchases['Quantity'].max()
print(f"Maximum quantity in a single purchase: {max_quantity}")

# Count unique categories
unique_categories = df_purchases['Category'].nunique()
print(f"Number of unique product categories: {unique_categories}")

# Value counts for 'Category'
category_counts = df_purchases['Category'].value_counts()
print("\nValue counts for 'Category':\n", category_counts)

# Group by 'Category' and calculate the sum of 'TotalAmount' for each category
sales_by_category = df_purchases.groupby('Category')['TotalAmount'].sum()
print("\nTotal sales by Category:\n", sales_by_category)

# Group by 'CustomerID' and calculate the total number of items purchased
items_by_customer = df_purchases.groupby('CustomerID')['Quantity'].sum()
print("\nTotal items purchased by CustomerID:\n", items_by_customer)

# Group by multiple columns and apply multiple aggregations
# For each CustomerID and Category, find the total quantity and average price
customer_category_summary = df_purchases.groupby(['CustomerID', 'Category']).agg(
    TotalQuantity=('Quantity', 'sum'),
    AveragePrice=('Price', 'mean')
)
print("\nCustomer and Category Summary:\n", customer_category_summary)
