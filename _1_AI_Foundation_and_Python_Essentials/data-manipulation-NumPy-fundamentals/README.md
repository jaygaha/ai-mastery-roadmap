# Practical Python for Data Manipulation: NumPy Fundamentals

While standard **Python** lists are versatile, they can become a significant bottleneck when dealing with the massive 
datasets common in artificial intelligence and machine learning. To perform numerical operations efficiently and 
effectively on large amounts of data, we turn to specialized libraries. **NumPy**, short for `Numerical Python`, is the 
foundational library for numerical computing in Python. It provides a powerful N-dimensional array object, **ndarray**, 
and sophisticated functions to operate on these arrays, offering substantial performance improvements over native 
Python lists—especially for numerical tasks. Understanding NumPy is not just about learning another Python library; 
it is about mastering the core technology that powers most data science and machine learning libraries like Pandas, 
Scikit-learn, and even deep learning frameworks such as TensorFlow and PyTorch. This lesson will equip you with the 
fundamental skills to manipulate and perform computations on numerical data, a critical step on your `AI development
roadmap`.

## Introduction to NumPy Arrays (`ndarrays`)

- The core of **NumPy** is the `ndarray` (n-dimensional array) object. 
- An `ndarray` is a grid of values, all the same type, indexed by a tuple of non-negative integers. The number of dimensions is called the array’s rank, while its shape is a tuple specifying the size along each dimension.

### Why NumPy Arrays Over Python Lists?

NumPy arrays offer several advantages over standard Python lists for numerical operations:

- **Performance**: NumPy arrays are implemented in C and Fortran, making them much faster for numerical operations compared to Python lists, which are collections of pointers to Python objects.
- **Memory Efficiency**: NumPy arrays store elements contiguously in memory. All elements must be of the same data type, leading to more efficient storage and access. Python lists, on the other hand, store references to objects, which can be scattered throughout memory, incurring overhead.
- **Functionality**: NumPy provides a vast collection of high-level mathematical functions to operate on arrays, including linear algebra routines, Fourier transforms, and random number capabilities, all optimized for performance.
- **Vectorization**: NumPy allows you to perform operations on entire arrays without explicitly writing for loops. This "vectorization" is not only more concise but also significantly faster due to optimized C implementations.

### Array Creation

**NumPy** provides various ways to create arrays. The most common method is using `np.array()`.

- From Python lists/tuples: `arr = np.array([1, 2, 3])`.
  - Example: [_1_init_array.py](_1_init_array.py)
- Using built-in functions:
  - **np.zeros(shape)**: `np.zeros((2, 3))` creates an array of zeros.
  - **np.ones(shape)**: `np.ones((2, 3))` creates an array of ones.
  - **np.full(shape, fill_value)**: Creates an array filled with a specified value.
  - **np.empty(shape)**: Creates an array whose initial content is random and depends on the state of the memory. This is faster than `zeros` or `ones` if you intend to fill the array elements later.
  - `np.identity(n)` or `np.eye(n)`: Creates a square N x N identity matrix (1s on the main diagonal, 0s elsewhere).
  - For creating arrays with sequences of numbers:
    - **np.arange(start, stop, step)**: `np.arange(0, 10, 2)` creates an array with evenly spaced values within a given range.
    - **np.linspace(start, stop, num)**: `np.linspace(0, 1, 5)` creates an array with evenly spaced values over a specified interval.

Example: [_2_func_array.py](_2_func_array.py)

#### Array Attributes

Every ndarray has several useful attributes that provide information about its structure:

- `ndarray.ndim`: The number of dimensions (axes) of the array.
- `ndarray.shape`: A tuple indicating the size of the array in each dimension.
- `ndarray.size`: The total number of elements in the array.
- `ndarray.dtype`: The data type of the elements in the array.

Example: [_3_attrib_array.py](_3_attrib_array.py)

#### Array Indexing and Slicing

Accessing and modifying elements within `NumPy` arrays is similar to Python lists but with extended capabilities for 
multiple dimensions.

##### Indexing

###### Basic Indexing
- **1D Arrays**: Similar to Python lists.
- **2D Arrays**: Use `array[row_index, column_index]` or `array[row_index][column_index]`. The first method is generally preferred for performance and readability.
- **N-D Arrays**: Extend this pattern `array[dim1_index, dim2_index, ..., dimN_index]`.

###### Boolean indexing

- `Boolean indexing` allows you to select elements from an array that satisfy a certain condition. 
- This is a powerful feature for filtering data.

##### Fancy Indexing

- `Fancy indexing` refers to using an array of indices to select elements or rows/columns. 
- It allows for non-contiguous and duplicate selections.

Example: [_4_1_indexing_array.py](_4_1_indexing_array.py)

##### Slicing

Slicing allows you to extract sub-arrays using the `start:stop:step` syntax, similar to Python lists, but applied across multiple dimensions.

Example: [_4_2_slicing_array.py](_4_2_slicing_array.py)

#### Array Manipulation

**NumPy** provides powerful tools for manipulating the shape and structure of arrays without changing the data itself.

##### Reshaping Arrays

- `reshape()` allows you to change the shape of an array. 
- The new shape must have the same total number of elements as the original array.
- `flatten()` always returns a copy, while `ravel()` returns a view of the original array if possible, otherwise a copy. `reshape(-1)` also often returns a view.

Example: [_5_1_reshaping_array.py](_5_1_reshaping_array.py)

##### Concatenating and Stacking Arrays

- `np.concatenate((arr1, arr2, ...), axis=0)`: Joins arrays along an existing axis.
- `np.vstack((arr1, arr2, ...))`: Stacks arrays vertically (row-wise). Equivalent to concatenate(..., axis=0) for 2D arrays.
- `np.hstack((arr1, arr2, ...))`: Stacks arrays horizontally (column-wise). Equivalent to concatenate(..., axis=1) for 2D arrays.

Arrays must have the same shape along the non-concatenation axis.

Example: [_5_2_concat_array.py](_5_2_concat_array.py)

##### Splitting Arrays

- `np.split(array, indices_or_sections, axis=0)`: Splits an array into multiple sub-arrays.
- `np.vsplit(array, indices_or_sections)`: Splits an array vertically.
- `np.hsplit(array, indices_or_sections)`: Splits an array horizontally.

Example: [_5_3_split_array.py](_5_3_split_array.py)

#### Mathematical Operations with NumPy Arrays

**NumPy's** true power lies in its ability to perform fast, element-wise mathematical operations on arrays.

##### Element-wise Operations

**Arithmetic operations** (`+`, `-`, `*`, `/`, `**`, `%`) are applied element-wise. This means operations are performed on corresponding elements of arrays.

Example: [_6_1_math_elementwise_array.py](_6_1_math_elementwise_array.py)

##### Broadcasting

Broadcasting is a powerful mechanism in **NumPy** that allows arrays with different shapes to be used in arithmetic operations. It works by "stretching" the smaller array across the larger array so that they have compatible shapes. The broadcasting rules are as follows:

- If the arrays do not have the same number of dimensions, the shape of the smaller array is padded with ones on its left side.
- If the shape of the two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.
- If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

> Broadcasting is essential for writing concise and efficient code in scientific computing, particularly in machine learning where operations often involve combining arrays of different, but compatible, shapes.

Example: [_6_2_math_broadcasting_array.py](_6_2_math_broadcasting_array.py)

##### Aggregate Functions

**NumPy** provides many functions for performing statistical aggregations on arrays: `sum()`, `mean()`, 
`std()` (standard deviation), `min()`, `max()`, `argmin()` (index of minimum), `argmax()` (index of maximum), `median()`, 
`percentile()`, etc.

- These can be applied to the entire array or along specific axes.

Example: [_6_3_math_agg_func_array.py](_6_3_math_agg_func_array.py)

##### Linear Algebra Operations

**NumPy** provides robust support for linear algebra, which is fundamental to many machine learning algorithms.

- **Dot Product**: For 1D arrays, it's the inner product. For 2D arrays (matrices), it's matrix multiplication. The @ operator (available in Python 3.5+) is a convenient shorthand for matrix multiplication.
- **Transpose**: `array.T` or `np.transpose(array)` swaps rows and columns.

Example: [_6_4_math_linear_alg_array.py](_6_4_math_linear_alg_array.py)


## Exercises

- [_7_exercise.py](_7_exercise.py)

## Conclusion

In this lesson, you learned the fundamentals of **NumPy**, the essential library for numerical computing in Python. We covered the `ndarray` object, its advantages over lists, and how to create and manipulate arrays using indexing, slicing, and boolean indexing. You also explored element-wise operations, broadcasting, aggregate functions, and basic linear algebra.
NumPy is crucial for data science and machine learning. Next, we’ll introduce Pandas, which builds on NumPy for advanced data analysis, preparing you for real-world tasks like customer churn prediction.
