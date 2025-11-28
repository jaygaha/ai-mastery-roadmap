import pandas as pd

# Example 4: Basic operations on Series
s1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['a', 'b', 'd']) # Note: different index for s2

print("\nSeries s1:\n", s1)
print("\nSeries s2:\n", s2)

# Addition - aligned by index
print("\nAddition (s1 + s2) - demonstrates index alignment:")
print(s1 + s2) # 'c' + 'd' results in NaN (Not a Number) because they don't align

# Scalar multiplication
print("\nScalar multiplication (s1 * 2):\n", s1 * 2)

# Filtering
print("\nFiltering s1 (values > 15):\n", s1[s1 > 15])