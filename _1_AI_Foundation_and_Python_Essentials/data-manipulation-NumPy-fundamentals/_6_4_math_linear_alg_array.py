import numpy as np

"""
Mathematical Operations with NumPy Arrays

    Linear Algebra Operations
"""

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")

# Matrix multiplication (dot product)
dot_product_AB = A.dot(B) # Using .dot() method
print(f"\nDot product (A.dot(B)):\n{dot_product_AB}")

dot_product_AB_alt = np.dot(A, B) # Using np.dot() function
print(f"Dot product (np.dot(A, B)):\n{dot_product_AB_alt}")

dot_product_AB_operator = A @ B # Using the @ operator (Python 3.5+)
print(f"Dot product (A @ B):\n{dot_product_AB_operator}")

# Transpose of a matrix
transpose_A = A.T
print(f"\nTranspose of A (A.T):\n{transpose_A}")

# Dot product of a vector and a matrix
v = np.array([9, 10]) # Shape (2,)
print(f"\nVector v: {v}")
dot_product_vA = v @ A # (1x2) @ (2x2) -> (1x2)
print(f"Dot product (v @ A): {dot_product_vA}") # Output: [39 58] (9*1+10*3, 9*2+10*4)

# Matrix inverse, determinant, etc., are also available in np.linalg
# inv_A = np.linalg.inv(A)
# print(f"\nInverse of A:\n{inv_A}")
# det_A = np.linalg.det(A)
# print(f"Determinant of A: {det_A}")

"""
Understanding these linear algebra operations is crucial as they form the backbone of many machine learning algorithms,
 from simple linear regression to complex neural network computations.
"""