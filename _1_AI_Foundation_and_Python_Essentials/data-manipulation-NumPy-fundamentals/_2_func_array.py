import numpy as np

"""
NumPy offers specialized functions for creating arrays with initial placeholder values:
    np.zeros(shape): Creates an array filled with zeros.
    np.ones(shape): Creates an array filled with ones.
    np.full(shape, fill_value): Creates an array filled with a specified value.
    np.empty(shape): Creates an array whose initial content is random and depends on the state of the memory. This is faster than zeros or ones if you intend to fill the array elements later.
    np.identity(n) or np.eye(n): Creates a square N x N identity matrix (1s on the main diagonal, 0s elsewhere).
"""

# Create an array of all zeros
zeros_array = np.zeros((2, 3)) # 2 rows, 3 columns
print("\nZeros Array:")
print(zeros_array)

print("\n")
print("-" * 30)
print("\n")

# Create an array of all ones
ones_array = np.ones((2, 3))  # 3 rows, 2 columns
print("Ones Array:")
print(ones_array)

print("\n")
print("-" * 30)
print("\n")

# Create an array filled with a specific value (e.g., 7)
full_array = np.full((2, 2), 7)
print("Full Array (filled with 7s):")
print(full_array)

print("\n")
print("-" * 30)
print("\n")

# Using np.empty (content will be arbitrary)
empty_array = np.empty((2, 2))
print("Empty Array (arbitrary content):")
print(empty_array)

print("\n")
print("-" * 30)
print("\n")

# Create an identity matrix
identity_matrix = np.eye(3) # 3x3 identity matrix
print("Identity Matrix:")
print(identity_matrix)

print("\n")
print("-" * 30)
print("\nSequences of numbers:")

# Using np.arange for a sequence
arange_array = np.arange(0, 10, 2) # Start at 0, up to (but not including) 10, step by 2
print("Arange Array:")
print(arange_array) # Output: [0 2 4 6 8]

# Using np.linspace for evenly spaced numbers
linespace_array = np.linspace(0, 1, 5) # 5 evenly spaced numbers between 0 and 1 (inclusive)
print("\nLinespace Array:")
print(linespace_array) # Output: [0.   0.25 0.5  0.75 1.  ]

"""
NumPy is excellent for generating arrays with random numbers, which is crucial for initializing weights in neural networks or simulating data.
    - np.random.rand(d0, d1, ..., dn): Creates an array of the given shape, filled with random samples from a uniform distribution over [0, 1).
    - np.random.randn(d0, d1, ..., dn): Creates an array of the given shape, filled with random samples from a standard normal (Gaussian) distribution (mean 0, variance 1).
    - np.random.randint(low, high, size): Returns random integers from low (inclusive) to high (exclusive), of the specified size.
"""

print("\n")
print("-" * 30)
print("\nGenerating arrays with random numbers:")

# Random numbers from a uniform distribution [0, 1)
rand_array = np.random.rand(2, 3)
print("Random Uniform Array:")
print(rand_array)

# Random numbers from a standard normal distribution
randn_array = np.random.randn(2, 2)
print("\nRandom Normal Array:")
print(randn_array)

# Random integers between 1 and 10 (exclusive of 10), in a 3x3 array
randint_array = np.random.randint(1, 10, size=(3, 3))
print("\nRandom Integer Array:")
print(randint_array)