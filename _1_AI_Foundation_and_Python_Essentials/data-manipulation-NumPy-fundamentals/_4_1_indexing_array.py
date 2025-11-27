import numpy as np

"""
Array Indexing

    Basic Indexing
"""

# 1D array indexing
arr_1d = np.arange(10, 20) # [10 11 12 13 14 15 16 17 18 19]
print(f"1D Array: {arr_1d}")
print(f"Element at index 3: {arr_1d[3]}") # Output: 13
arr_1d[3] = 99 # Modifying an element
print(f"Modified 1D Array: {arr_1d}")

print("-" * 100)

# 2D array indexing
arr_2d = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
print(f"2D Array:\n{arr_2d}")
print(f"Element at row 1, column 2: {arr_2d[1, 2]}") # Output: 15
print(f"Element at row 0, column 0: {arr_2d[0][0]}") # Output: 10 (alternative syntax)

print("-" * 100)

# Modifying a 2D array element
arr_2d[0, 1] = 100
print(f"Modified 2D Array:\n{arr_2d}")

# Accessing an entire row or column
print(f"First row: {arr_2d[0, :]}") # Access all columns in the first row
print(f"Last column: {arr_2d[:, 2]}") # Access all rows in the last column
print(f"Second row: {arr_2d[1]}") # When a single index is given for the first dimension, it returns that entire row

print("\n")
print("-" * 100)
print("\n")

"""
    Boolean Indexing
"""

print("Boolean Indexing\n")

data = np.array([10, 20, 30, 40, 50, 60])
print(f"Original Data: {data}")

# Create a boolean array based on a condition
condition = (data > 30)
print(f"Boolean condition array: {condition}") # Output: [False False False  True  True  True]

# Use the boolean array to select elements
filtered_data = data[condition]
print(f"Filtered data (elements > 30): {filtered_data}") # Output: [40 50 60]

# Combine conditions (use & for AND, | for OR)
combined_condition = (data > 20) & (data < 50)
filtered_combined = data[combined_condition]
print(f"Filtered data (elements > 20 AND < 50): {filtered_combined}") # Output: [30 40]

# Boolean indexing on a 2D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(f"\nOriginal Matrix:\n{matrix}")

# Select elements greater than 5
greater_than_5 = matrix[matrix > 5]
print(f"Elements greater than 5: {greater_than_5}") # Output: [6 7 8 9] (returns a 1D array)

# Set elements based on a condition
matrix[matrix % 2 == 0] = 0 # Set even numbers to 0
print(f"Matrix with even numbers set to 0:\n{matrix}")

print("\n")
print("-" * 100)
print("\n")

"""
    Fancy Indexing
"""

print("Fancy Indexing\n")

arr = np.arange(0, 100, 10) # [ 0 10 20 30 40 50 60 70 80 90]
print(f"Original 1D array: {arr}")

# Select elements at specific indices
indices = [1, 5, 8]
fancy_indexed_1d = arr[indices]
print(f"Elements at indices {indices}: {fancy_indexed_1d}") # Output: [10 50 80]

# Select elements in a specific order, potentially with duplicates
custom_order_indices = [3, 0, 3, 7]
custom_ordered_arr = arr[custom_order_indices]
print(f"Elements in custom order {custom_order_indices}: {custom_ordered_arr}") # Output: [30  0 30 70]

print("-" * 100)

# Fancy indexing on a 2D array
matrix = np.array([[10, 11, 12],
                   [13, 14, 15],
                   [16, 17, 18],
                   [19, 20, 21]])
print(f"Original 2D matrix:\n{matrix}")

# Select specific rows
row_indices = [0, 2]
selected_rows = matrix[row_indices]
print(f"Selected rows (0 and 2):\n{selected_rows}")

# Select specific rows and specific columns (tuple of index arrays)
# This selects elements (0, 0), (1, 2), (2, 1)
selected_elements = matrix[[0, 1, 2], [0, 2, 1]]
print(f"Selected elements (0,0), (1,2), (2,1): {selected_elements}") # Output: [10 15 17]
