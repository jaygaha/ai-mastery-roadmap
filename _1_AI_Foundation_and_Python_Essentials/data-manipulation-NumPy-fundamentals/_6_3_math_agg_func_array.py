import numpy as np

"""
Mathematical Operations with NumPy Arrays

    Aggregate Functions
"""

data_matrix = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
print(f"Data Matrix:\n{data_matrix}")

# Sum of all elements in the array
print(f"\nSum of all elements: {np.sum(data_matrix)}") # Output: 45

# Sum along axis 0 (columns) - sums down each column
print(f"Sum along axis 0 (columns): {np.sum(data_matrix, axis=0)}") # Output: [12 15 18] (sum of [1,4,7], [2,5,8], [3,6,9])

# Sum along axis 1 (rows) - sums across each row
print(f"Sum along axis 1 (rows): {np.sum(data_matrix, axis=1)}") # Output: [ 6 15 24] (sum of [1,2,3], [4,5,6], [7,8,9])

# Mean of all elements
print(f"\nMean of all elements: {np.mean(data_matrix)}")

# Max value along axis 0
print(f"Max value along axis 0: {np.max(data_matrix, axis=0)}") # Output: [7 8 9]

# Index of max value along axis 1
print(f"Index of max value along axis 1: {np.argmax(data_matrix, axis=1)}") # Output: [2 2 2] (index 2 for each row)

# Standard deviation
print(f"Standard deviation of all elements: {np.std(data_matrix)}")
