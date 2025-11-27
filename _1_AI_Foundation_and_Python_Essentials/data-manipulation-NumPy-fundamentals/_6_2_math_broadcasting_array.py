import numpy as np

"""
Mathematical Operations with NumPy Arrays

    Broadcasting
"""

A = np.array([[1, 2, 3], [4, 5, 6]]) # Shape (2, 3)
b = np.array([10, 20, 30])           # Shape (3,)

print(f"Matrix A:\n{A}")
print(f"Vector b: {b}")

# When A + b is performed, b (shape (3,)) is broadcasted across A's rows.
# Conceptually, b becomes [[10, 20, 30], [10, 20, 30]] for the operation.
result = A + b
print(f"\nResult of A + b:\n{result}")

print("-" * 100)

# Another example: Adding a column vector to a matrix
C = np.array([[10, 20, 30],
              [40, 50, 60]]) # Shape (2, 3)
d = np.array([[1],
              [2]])          # Shape (2, 1) - explicit column vector

print(f"Matrix C:\n{C}")
print(f"Column vector d:\n{d}")

# When C + d is performed, d (shape (2,1)) is broadcasted across C's columns.
# Conceptually, d becomes [[1, 1, 1], [2, 2, 2]] for the operation.
result_cd = C + d
print(f"\nResult of C + d:\n{result_cd}")

print("-" * 100)

# Example: Multiplying a 2D array by a scalar (this is also broadcasting)
matrix = np.array([[1, 2], [3, 4]])
scalar = 5
product = matrix * scalar # Scalar 5 is broadcasted to [[5,5],[5,5]]
print(f"Matrix * Scalar:\n{product}")