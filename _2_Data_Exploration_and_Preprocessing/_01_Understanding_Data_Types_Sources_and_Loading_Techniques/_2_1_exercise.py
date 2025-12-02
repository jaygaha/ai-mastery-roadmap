"""
1. Loading from a Custom Delimiter File:

Create a text file named _2_1_product_ratings.txt with the following content, where fields are separated by pipes (|):
```javascript
ProductID|ProductName|Category|Rating|ReviewDate
P101|Laptop Pro|Electronics|4.5|2025-01-20
P102|Smartphone X|Electronics|3.8|2025-02-15
P103|Smartwatch S|Wearable|4.9|2025-03-01
P104|Wireless Buds|Audio||2025-03-10
```

Load this file into a Pandas DataFrame.
Ensure ReviewDate is parsed as a datetime object.
Check the data types (dtypes) of the resulting DataFrame. What data type did Rating get? Why? (Hint: Notice the missing value for P104).
How would you modify the pd.read_csv() call to ensure Rating is a float64 from the start, even with the missing value? (Hint: consider na_values).
"""

import pandas as pd
import io

# 1. Create the text file content in a string for demonstration
file_content = """ProductID|ProductName|Category|Rating|ReviewDate
P101|Laptop Pro|Electronics|4.5|2025-01-20
P102|Smartphone X|Electronics|3.8|2025-02-15
P103|Smartwatch S|Wearable|4.9|2025-03-01
P104|Wireless Buds|Audio||2025-03-10"""

# Simulate saving to a file named '_2_1_product_ratings.txt'
file_path = '_2_1_product_ratings.txt'
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(file_content)

print(f"Created file: {file_path}\n")

# --- Part A: Load the file with default settings and check dtypes ---
print("--- Part A: Initial load ---")

# Use sep='|' for the pipe delimiter and parse 'ReviewDate' as datetime
df_initial = pd.read_csv(
    file_path,
    sep='|',
    parse_dates=['ReviewDate']
)

print("DataFrame dtypes (initial):\n", df_initial.dtypes)
print("\nDataFrame content (initial):\n", df_initial)

# Explanation of Rating dtype in Part A:
print("\n--- Explanation ---")
print("In the initial load, the 'Rating' column is read as 'object' (string).")
print("This happens because the missing value for product P104 (Audio||2025-03-10) is interpreted as an empty string ('').")
print("A column containing a mix of numbers and strings must be stored using the generic 'object' dtype in standard Pandas configurations.")

# --- Part B: Modify the read_csv call to ensure Rating is float64 ---
print("\n--- Part B: Modified load to force float64 Rating ---")

# Use the na_values parameter to explicitly tell pandas that empty strings in the 'Rating'
# column should be treated as NaN (Not a Number), allowing the entire column to be float.
df_modified = pd.read_csv(
    file_path,
    sep='|',
    parse_dates=['ReviewDate'],
    na_values={'Rating': ['']} # Treat empty strings in 'Rating' column as NaN
)

print("DataFrame dtypes (modified):\n", df_modified.dtypes)
print("\nDataFrame content (modified):\n", df_modified)

# Clean up the created file (optional)
import os
os.remove(file_path)
print(f"\nCleaned up file: {file_path}")