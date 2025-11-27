"""
Exercise 1: Array Creation and Attributes

    Create a 1D NumPy array named data_1d containing integers from 0 to 9.
    Print its shape, ndim, and dtype.
    Create a 2D NumPy array named matrix_5x3 with 5 rows and 3 columns, filled with random floating-point numbers between 0 and 1.
    Print its shape, ndim, and dtype.
    Create a 3x3 identity matrix named identity_matrix.
"""

import numpy as np

print("\nExercise 1: Array Creation and Attributes")

# Create a 1D NumPy array named data_1d containing integers from 0 to 9.
# data_1d = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
data_1d = np.arange(10) # Using arange() is slightly more idiomatic

# Print its shape, ndim, and dtype.
print(f"Type of array_1d: {type(data_1d)}") # Output: <class 'numpy.ndarray'>
print(f"Shape of array_1d: {data_1d.shape}") # Output: (10,) - A tuple indicating 5 elements
print(f"Number of dimensions: {data_1d.ndim}") # Output: 1
print(f"data_1d data type (dtype): {data_1d.dtype}") # int64

# Create a 2D NumPy array named matrix_5x3 with 5 rows and 3 columns, filled with random floating-point numbers between 0 and 1.
matrix_5x3 = np.random.rand(5, 3)
print(matrix_5x3)

# Print its shape, ndim, and dtype.
print(f"Shape of matrix_5x3: {matrix_5x3.shape}")
print(f"Number of dimensions: {matrix_5x3.ndim}")
print(f"Type of matrix_5x3: {type(matrix_5x3)}") # Output: <class 'numpy.ndarray'>
print(f"matrix_5x3 data type (dtype): {matrix_5x3.dtype}") #  float64

# Create a 3x3 identity matrix named identity_matrix.
identity_matrix = np.identity(3)
print(identity_matrix)


print("-" * 100)

"""
Exercise 2: Indexing and Slicing

    Given the following 2D array:
    
    grid = np.array([[10, 11, 12, 13],
                     [14, 15, 16, 17],
                     [18, 19, 20, 21],
                     [22, 23, 24, 25]])
    
    Select the element 20 using basic indexing.
    Extract the sub-array [[15, 16], [19, 20]] using slicing.
    Extract the entire second column (containing [11, 15, 19, 23]).
    Using boolean indexing, select all elements in grid that are greater than 18.
    Using fancy indexing, select rows 0 and 3.
"""

print("\nExercise 2: Indexing and Slicing")

grid = np.array([[10, 11, 12, 13],
                     [14, 15, 16, 17],
                     [18, 19, 20, 21],
                     [22, 23, 24, 25]])

print(f"Grid array:\n {grid}")

# Select the element 20 using basic indexing.
print(f"Element: {grid[2, 2]}") # Output: 20

# Extract the sub-array [[15, 16], [19, 20]] using slicing.
# Rows 1 to 3 (exclusive), Columns 1 to 3 (exclusive)
sub_array = grid[1:3, 1:3]
print(f"Sub-array (15, 16, 19, 20):\n{sub_array}")

# Extract the entire second column (containing [11, 15, 19, 23]).
# Use a colon : to select all rows, and index 1 for the second column
print(f"Column 2: {grid[:, 1]}")

# Using boolean indexing, select all elements in grid that are greater than 18.
condition = (grid > 18)
filtered_data = grid[condition]
print(f"Boolean condition array: {filtered_data}")

#  Using fancy indexing, select rows 0 and 3.
row_indices = [0, 3]
selected_rows = grid[row_indices]
print(f"Selected rows (0 and 3):\n{selected_rows}")

print("-" * 100)

"""
Exercise 3: Array Manipulation

Given the following 1D array:
python

values = np.arange(20) # [0, 1, ..., 19]

    Reshape values into a 4x5 matrix named reshaped_matrix.
    Flatten reshaped_matrix back into a 1D array using ravel().
    Create two 2x2 matrices: A = np.array([[1,2],[3,4]]) and B = np.array([[5,6],[7,8]]).
    Vertically stack A and B to create a 4x2 matrix.
    Horizontally stack A and B to create a 2x4 matrix.
"""

print("\nExercise 3: Array Manipulation")

values = np.arange(20)

print(f"values: {values}") # [0, 1, ..., 19]

# Reshape values into a 4x5 matrix named reshaped_matrix.
reshaped_4x5 = values.reshape(4, 5)

print(f"reshaped_4x5: \n{reshaped_4x5}")

# Flatten reshaped_matrix back into a 1D array using ravel().
flattened = reshaped_4x5.ravel()
print(f"\nFlattened array:\n{flattened}")
print(f"Shape: {flattened.shape}")

# Create two 2x2 matrices: A = np.array([[1,2],[3,4]]) and B = np.array([[5,6],[7,8]]).
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

print(f"A:\n{A}")
print(f"B:\n{B}")

# Vertically stack A and B to create a 4x2 matrix.
vstack_arr = np.vstack((A, B))
print(f"\nVertical Stack:\n{vstack_arr}")

# Horizontally stack A and B to create a 2x4 matrix.
hstack_arr = np.hstack((A, B))
print(f"\nHorizontal Stack:\n{hstack_arr}")

print("-" * 100)

"""
Exercise 4: Mathematical Operations

Given the following arrays:

    m1 = np.array([[1, 2], [3, 4]])
    m2 = np.array([[5, 6], [7, 8]])
    v = np.array([10, 20])

    Perform element-wise addition of m1 and m2.
    Perform element-wise multiplication of m1 and m2.
    Calculate the matrix product of m1 and m2 using the @ operator.
    Add the vector v to each row of m1 using broadcasting.
    Calculate the sum of all elements in m1.
    Calculate the mean of each column in m2.
"""

print("\nExercise 4: Mathematical Operations")

m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])
v = np.array([10, 20])

print(f"m1: \n{m1}")
print(f"m2: \n{m2}")
print(f"v: \n{v}")

# Perform element-wise addition of m1 and m2.
print(f"\nAddition (m1 + m2):\n{m1 + m2}")

# Perform element-wise multiplication of m1 and m2.
print(f"\nMultiplication (m1 * m2):\n{m1 * m2}")

# Calculate the matrix product of m1 and m2 using the @ operator.
print(f"Dot product (m1 @ m2):\n{m1 @ m2}")

# Add the vector v to each row of m1 using broadcasting.
result = m1 + v
print(f"\nResult of m1 + v:\n{result}")

# Calculate the sum of all elements in m1.
print(f"\nSum of all elements: {np.sum(m1)}") # Output: 10

# Calculate the mean of each column in m2.
column_means = np.mean(m2, axis=0)
print(f"\nMean of each column in m2: {column_means}")

print("-" * 100)