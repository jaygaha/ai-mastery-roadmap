import numpy as np

"""
Array Slicing

Slicing allows you to extract sub-arrays using the start:stop:step syntax, similar to Python lists, but applied across multiple dimensions.
"""

arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])
print(f"Original Array:\n{arr}")

# Slice rows 0 and 1, and columns 1 and 2
slice_1 = arr[:2, 1:3] # rows from start up to (but not including) index 2, columns from index 1 up to (not including) 3
print(f"\nSlice 1 (rows 0-1, cols 1-2):\n{slice_1}") # Output: [[ 2  3] [ 6  7]]

print("-" * 100)

# Get the last two rows
slice_2 = arr[1:, :]
print(f"\nSlice 2 (last two rows):\n{slice_2}") # Output: [[ 5  6  7  8] [ 9 10 11 12]]

print("-" * 100)

# Get every other element from the second column (step)
slice_3 = arr[::2, 1] # rows starting from 0, step by 2; column 1
print(f"\nSlice 3 (every other row from column 1):\n{slice_3}") # Output: [ 2 10]

print("-" * 100)

# Slicing creates a *view* into the original array, not a copy.
# Modifying a slice will modify the original array.
slice_1[0, 0] = 999
print(f"\nOriginal Array after modifying slice_1:\n{arr}") # Notice arr[0, 1] is now 999

print("-" * 100)

# If you need a copy, use the .copy() method
arr_copy = arr[:2, 1:3].copy()
arr_copy[0, 0] = 111
print(f"\nOriginal Array after modifying arr_copy (no change):\n{arr}")
print(f"Copied Slice:\n{arr_copy}")