import numpy as np

"""
Array Manipulation

    Splitting Arrays
"""

arr = np.arange(16).reshape(4, 4)
print(f"Original Array to split:\n{arr}")

# Split into 2 equal parts along axis 0 (rows)
split_rows = np.split(arr, 2, axis=0)
print(f"\nSplit into 2 equal parts (rows):\nPart 1:\n{split_rows[0]}\nPart 2:\n{split_rows[1]}")

# Split into 4 equal parts along axis 1 (columns)
split_cols = np.split(arr, 4, axis=1)
print(f"\nSplit into 4 equal parts (columns):\nPart 1:\n{split_cols[0]}\nPart 2:\n{split_cols[1]}\nPart 3:\n{split_cols[2]}\nPart 4:\n{split_cols[3]}")

# Split at specific indices (rows)
split_at_indices = np.split(arr, [1, 3], axis=0) # Splits before index 1, then before index 3
print(f"\nSplit at indices [1, 3] (rows):\nPart 1 (row 0):\n{split_at_indices[0]}\nPart 2 (rows 1-2):\n{split_at_indices[1]}\nPart 3 (row 3):\n{split_at_indices[2]}")

# Using vsplit and hsplit for convenience
vsplit_arr = np.vsplit(arr, 2)
print(f"\nVsplit into 2 equal parts:\nPart 1:\n{vsplit_arr[0]}\nPart 2:\n{vsplit_arr[1]}")

hsplit_arr = np.hsplit(arr, [2]) # Split before column index 2
print(f"\nHsplit at index 2:\nPart 1:\n{hsplit_arr[0]}\nPart 2:\n{hsplit_arr[1]}")