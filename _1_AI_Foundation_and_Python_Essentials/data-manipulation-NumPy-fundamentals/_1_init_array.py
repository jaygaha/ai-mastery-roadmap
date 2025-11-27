import numpy as np

# Creating a 1-dimensional array (vector) from a Python list
list_1d = [1, 2, 3, 4, 5]
array_1d = np.array(list_1d)
print("1D Array:")
print(array_1d)
print(f"Type of array_1d: {type(array_1d)}") # Output: <class 'numpy.ndarray'>
print(f"Shape of array_1d: {array_1d.shape}") # Output: (5,) - A tuple indicating 5 elements
print(f"Number of dimensions: {array_1d.ndim}") # Output: 1

print("-" * 30)
print("\n")

# Creating a 2-dimensional array (matrix) from a nested Python list
list_2d = [[1, 2, 3], [4, 5, 6]]
array_2d = np.array(list_2d)
print("2D Array:")
print(array_2d)
print(f"Shape of array_2d: {array_2d.shape}") # Output: (2, 3) - 2 rows, 3 columns
print(f"Number of dimensions: {array_2d.ndim}") # Output: 2

print("-" * 30)
print("\n")

# Creating a 3-dimensional array (tensor)
list_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
array_3d = np.array(list_3d)
print("3D Array:")
print(array_3d)
print(f"Shape of array_3d: {array_3d.shape}") # Output: (2, 2, 2) - 2 "layers", each 2 rows, 2 columns
print(f"Number of dimensions: {array_3d.ndim}") # Output: 3