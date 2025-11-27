import numpy as np

"""
Array Attributes
"""

# Example array
arr = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
print(f"Array:\n{arr}")
print(f"Number of dimensions (ndim): {arr.ndim}")
print(f"Shape of array: {arr.shape}") # (rows, columns)
print(f"Total number of elements (size): {arr.size}")
print(f"Data type of elements (dtype): {arr.dtype}") # Output: float64 (default for decimals)

print("-" * 50)

# Specifying data type explicitly
int_array = np.array([1, 2, 3], dtype=np.int32)
print(f"\nInteger Array with specific dtype: {int_array}")
print(f"Data type: {int_array.dtype}") # Output: int32


print("-" * 50)

# Changing data type (type casting)
float_array = int_array.astype(np.float64)
print(f"\nCasted to Float Array: {float_array}")
print(f"Data type: {float_array.dtype}") # Output: float64

"""
Choosing the correct data type (dtype) is important for memory efficiency and computational accuracy. Common data types
include int8, int16, int32, int64 (integers of various sizes), float16, float32, float64 (floating-point numbers), 
and bool (boolean values).
"""