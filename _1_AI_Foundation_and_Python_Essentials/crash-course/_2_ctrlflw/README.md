# Control Flow

Control flow statements determine the order in which individual statements or instructions are executed in a program. 
They are fundamental for creating dynamic and responsive applications that can make decisions and perform 
repetitive tasks.

## Conditional Statements: `if`, `elif`, `else`

Conditional statements allow your program to execute different blocks of code based on whether certain conditions 
are true or false.

### Basic `if` Statement

An `if` statement evaluates a condition. If the condition is `True`, the indented block of code following the `if` 
statement is executed.
```
if condition is true:
    operation
```

### `if-else` Statement

The `if-else` statement provides an alternative block of code to execute if the `if` condition is `False`.
```
if condition is true:
    operation
else:
    default
```

### `if-elif-else` Statement

For multiple conditions, you can use `elif` (short for "else if"). Python will check conditions sequentially and 
execute the block corresponding to the first `True` condition. If none of the `if` or `elif` conditions are met, the 
`else` block (if present) is executed.
```
if condition1 is true:
    operation
elif condition2 is true:
    operation
else:
    default
```

### Logical Operators (`and`, `or`, `not`)

You can combine multiple conditions using logical operators:
- `and`: Both conditions must be True.
- `or`: At least one condition must be True.
- `not`: Reverses the truth value of a condition.

```
if conditon is true AND condition1 is true
    operation
    
if conditon is true OR condition1 is true
    operation

if not conditon is true
    operation
```

### Example

Check this example implementation [_1_if.py](_1_if.py)

## Loops

In Python, loops are used to repeatedly execute a block of code until a specific condition is met, or for each item in 
a sequence. They are essential for automating repetitive tasks, processing large datasets, and improving code efficiency
by avoiding repetition (adhering to the "Don't Repeat Yourself" (DRY) principle). Python offers two primary types of loops:
the `for` loop and the `while` loop.

### 1. `for` Loops

`for` loops are used for iterating over a sequence (like a list, tuple, string, or range) or other iterable objects. 
They are ideal when you know how many times you want to loop or when you need to process each item in a collection.

```
for variable in sequence:
    # Code block to execute for each item
```

#### The `range()` Function

`range()` generates a sequence of numbers, which is often used in `for` loops to iterate a specific number of times.

- `range(stop)`: Generates numbers from 0 up to `stop-1`.
- `range(start, stop)`: Generates numbers from `start` up to `stop-1`.
- `range(start, stop, step)`: Generates numbers from `start` up to `stop-1`, incrementing by `step`.

```
for i in range(4):
    print(i)
```

### 2. `while` Loops

A `while` loop is used to execute a block of code as long as a given condition remains True. It is ideal for scenarios
where the number of iterations is not known in advance

```
while condition:
    # Code block to execute
```

## Loop Control Statements: `break` and `continue`

Python offers control statements to alter the normal flow of a loop: 
- `break`: Terminates the entire loop immediately and transfers execution to the statement immediately following the loop.
- `continue`: Skips the remainder of the current iteration's code block and jumps to the next iteration of the loop.
- `pass`: A null operation; nothing happens when it executes. It is used as a placeholder where a statement is syntactically required, but you don't want any code to execute yet.

## Nested Loops

You can also place a loop inside another loop, known as a nested loop. This is useful for working with `multi-dimensional` 
data structures like matrices or grids.
```
for i in range(2):
    for j in range(2):
        print(f"i = {i}, j = {j}")
```

### Example

Check this example implementation [_2_loops.py](_2_loops.py)

## Conclusion

**Core Concepts Covered**

The lesson introduced Python’s essential data structures: `lists`, `tuples`, `dictionaries`, and `sets`. It detailed how to create, access, and modify each, emphasizing their differences in mutability, order, and uniqueness. Control flow mechanisms—such as `if-elif-else` for decision-making and `for`/`while` loops for repetition—were also explored, including the use of `break` and `continue` for loop control.

**Significance**

These concepts form the foundation for advanced AI and machine learning. Mastery of data manipulation and conditional logic is crucial for building dynamic, data-driven applications.

**Next Steps**

The following lessons will apply these fundamentals:

- **NumPy Fundamentals**: Focuses on numerical operations for large datasets, leveraging list principles.
- **Pandas Introduction**: Covers DataFrames, which are enhanced tables used for data loading, cleaning, and analysis.
- **Case Study**: Skills from this lesson will be used in the Customer Churn Prediction Case Study.