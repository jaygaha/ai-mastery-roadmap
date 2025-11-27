


arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print(f"Array 1:\n{arr1}")
print(f"Array 2:\n{arr2}")

# Addition
print(f"\nAddition (arr1 + arr2):\n{arr1 + arr2}")

# Subtraction
print(f"\nSubtraction (arr1 - arr2):\n{arr1 - arr2}")
# Multiplication (element-wise)
print(f"\nElement-wise Multiplication (arr1 * arr2):\n{arr1 * arr2}")
# Division
print(f"\nDivision (arr1 / arr2):\n{arr1 / arr2}")
# Exponentiation
print(f"\nExponentiation (arr1 ** 2):\n{arr1 ** 2}")

# You can also perform operations with a scalar
print(f"\nArray 1 + 10:\n{arr1 + 10}")
print(f"\nArray 1 * 3:\n{arr1 * 3}")

print("\n")
print("*" * 100)
print("\n")
"""
NumPy also provides universal functions (ufuncs) for common mathematical operations: 
    np.add(), np.subtract(), np.multiply(), np.divide(), np.sqrt(), np.exp(), np.log(), np.sin(), np.cos(), etc. 
These are typically faster for complex operations.
"""

print(f"Universal functions")
x = np.array([0, np.pi/2, np.pi])
print(f"x: {x}")
print(f"sin(x): {np.sin(x)}") # Output: [0.00000000e+00 1.00000000e+00 1.22464680e-16] (close to 0 due to float precision)

y = np.array([1, 4, 9])
print(f"\ny: {y}")
print(f"sqrt(y): {np.sqrt(y)}")
