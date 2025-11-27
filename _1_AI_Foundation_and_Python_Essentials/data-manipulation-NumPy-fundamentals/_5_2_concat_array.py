import numpy as np

"""
Array Manipulation

    Concatenating and Stacking Arrays
"""

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print(f"Array 1:\n{arr1}")
print(f"Array 2:\n{arr2}")

# Concatenate along axis 0 (rows)
concat_axis0 = np.concatenate((arr1, arr2), axis=0)
print(f"\nConcatenate along axis 0 (rows):\n{concat_axis0}")

# Concatenate along axis 1 (columns)
concat_axis1 = np.concatenate((arr1, arr2), axis=1)
print(f"\nConcatenate along axis 1 (columns):\n{concat_axis1}")

# Vertical stacking (vstack)
vstack_arr = np.vstack((arr1, arr2))
print(f"\nVertical Stack:\n{vstack_arr}")

# Horizontal stacking (hstack)
hstack_arr = np.hstack((arr1, arr2))
print(f"\nHorizontal Stack:\n{hstack_arr}")

# Stacking a 1D array with a 2D array (requires careful dimension handling)
arr_1d_row = np.array([9, 10]).reshape(1, -1) # Make it a 2D row vector (1, 2)
vstack_with_1d = np.vstack((arr1, arr_1d_row))
print(f"\nVertical Stack with a 1D row vector:\n{vstack_with_1d}")