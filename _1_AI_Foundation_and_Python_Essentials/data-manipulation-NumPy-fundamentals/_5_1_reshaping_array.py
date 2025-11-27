import numpy as np

"""
Array Manipulation

    Reshaping Arrays
"""

arr_1d = np.arange(12) # [ 0  1  2  3  4  5  6  7  8  9 10 11]
print(f"Original 1D array:\n{arr_1d}")
print(f"Shape: {arr_1d.shape}")

# Reshape to a 3x4 matrix
reshaped_3x4 = arr_1d.reshape(3, 4)
print(f"\nReshaped to 3x4:\n{reshaped_3x4}")
print(f"Shape: {reshaped_3x4.shape}")

# Use -1 to automatically infer one dimension
reshaped_2_auto = arr_1d.reshape(2, -1) # 2 rows, columns inferred (6)
print(f"\nReshaped to 2x (inferred):\n{reshaped_2_auto}")
print(f"Shape: {reshaped_2_auto.shape}")

reshaped_auto_3 = arr_1d.reshape(-1, 3) # (inferred) rows, 3 columns
print(f"\nReshaped to x3 (inferred):\n{reshaped_auto_3}")
print(f"Shape: {reshaped_auto_3.shape}")

# Reshape to a 1D array (flattening)
flattened = reshaped_3x4.reshape(-1) # or reshaped_3x4.flatten() or reshaped_3x4.ravel()
print(f"\nFlattened array:\n{flattened}")
print(f"Shape: {flattened.shape}")