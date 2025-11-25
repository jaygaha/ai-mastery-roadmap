# 1. LISTS
# -> ordered, mutable collections of items,
#   meaning it can be changed their contents after creation & maintaining their position
# -> defined by enclosing comma-separated elements within square brackets []

# An empty list
empty_list = []
print(f"Empty list: {empty_list}")

# A list of integers (homogeneous)
ages = [25, 30, 22, 35, 28]
print(f"List of ages: {ages}")

# A list of strings (homogeneous)
names = ["Jay", "Akita", "Ishi", "Yama"]
print(f"List of names: {names}")

# A list with mixed data types (heterogeneous) - common in data processing
customer_record = ["Yamagawa", 30, True, 1299.50] # Name, Age, Active, Balance
print(f"Customer record list: {customer_record}")

#
# Accessing Elements: Indexing and Slicing
# -> Elements in a list are accessed using their index, which starts from 0 for the first element.
# -> Python supports negative indexing, where -1 refers to the last element, -2 to the second to last, and so on
# Slicing allows to extract a sub-sequence of elements

print("\nAccessing Elements: Indexing and Slicing")

data_points = [10, 20, 30, 40, 50, 60, 70]

print(f"Data points: {data_points}")

# Accessing a single element by positive index
first_element = data_points[0]
print(f"First element: {first_element}") # Output: 10

third_element = data_points[2]
print(f"Third element: {third_element}") # Output: 30

# Accessing a single element by negative index
last_element = data_points[-1]
print(f"Last element: {last_element}") # Output: 70

second_to_last = data_points[-2]
print(f"Second to last element: {second_to_last}") # Output: 60

# Slicing: [start:end] - end index is exclusive
sub_list1 = data_points[1:4] # Elements from index 1 up to (but not including) index 4
print(f"Slice [1:4]: {sub_list1}") # Output: [20, 30, 40]

# Slicing from the beginning: [:end]
sub_list2 = data_points[:3] # Elements from start up to (but not including) index 3
print(f"Slice [:3]: {sub_list2}") # Output: [10, 20, 30]

# Slicing to the end: [start:]
sub_list3 = data_points[4:] # Elements from index 4 to the end
print(f"Slice [4:]: {sub_list3}") # Output: [50, 60, 70]

# Slicing with a step: [start:end:step]
every_other_element = data_points[0::2] # Start at 0, go to end, step by 2
print(f"Every other element: {every_other_element}") # Output: [10, 30, 50, 70]

# Reversing a list using slicing
reversed_list = data_points[::-1]
print(f"Reversed list: {reversed_list}") # Output: [70, 60, 50, 40, 30, 20, 10]

#
# Modifying Lists
# -> Since lists are mutable, yuo can change individual elements, add new elements, or remove

print("\nModifying Lists")

sensor_readings = [23.5, 24.1, 22.9, 25.0]
print(f"Sensor readings: {sensor_readings}")

# Modifying an element
sensor_readings[1] = 24.0 # Correcting a reading at index 1
print(f"Modified readings in index 1: {sensor_readings}") # Output: [23.5, 24.0, 22.9, 25.0]

# Adding elements: append() adds to the end
sensor_readings.append(26.2)
print(f"After append: {sensor_readings}") # Output: [23.5, 24.0, 22.9, 25.0, 26.2]

# Adding elements: insert(index, item) adds at a specific position
sensor_readings.insert(0, 23.0) # Insert at the beginning
print(f"After insert: {sensor_readings}") # Output: [23.0, 23.5, 24.0, 22.9, 25.0, 26.2]

# Extending a list with another list (or any iterable)
new_readings = [27.1, 28.3]
sensor_readings.extend(new_readings)
print(f"After extend: {sensor_readings}") # Output: [23.0, 23.5, 24.0, 22.9, 25.0, 26.2, 27.1, 28.3]

# Removing elements: remove(value) removes the first occurrence of a value
sensor_readings.remove(22.9)
print(f"After remove 22.9: {sensor_readings}") # Output: [23.0, 23.5, 24.0, 25.0, 26.2, 27.1, 28.3]

# Removing elements: pop(index) removes and returns element at a specific index (defaults to last)
popped_value = sensor_readings.pop() # Removes last element
print(f"Popped value: {popped_value}, List after pop: {sensor_readings}") # Output: Popped value: 28.3, List after pop: [23.0, 23.5, 24.0, 25.0, 26.2, 27.1]

# Deleting elements by index or slice using del statement
del sensor_readings[0] # Deletes element at index 0
print(f"After del index 0: {sensor_readings}") # Output: [23.5, 24.0, 25.0, 26.2, 27.1]

# Other useful list methods
sensor_readings.sort() # Sorts the list in place (ascending)
print(f"Sorted list: {sensor_readings}")

sensor_readings.reverse() # Reverses the list in place
print(f"Reversed sorted list: {sensor_readings}")

# Clearing all elements
sensor_readings.clear()
print(f"After clear: {sensor_readings}")

#
# List Comprehensions
# -> a concise and elegant way to create lists.
# -> provide a compact syntax for creating a new list from an existing sequence,
#       often replacing more verbose for loop constructions.
#
print("\nList Comprehension")

# Example 1: Creating a list of squares
numbers = [1, 2, 3, 4, 5]
print(f"List comprehension example: {numbers}")

# Using a for loop (traditional way)
squares_loop = []
for num in numbers:
    squares_loop.append(num ** 2)
print(f"Squares (loop): {squares_loop}")

# Using a list comprehension
squares_comprehension = [num ** 2 for num in numbers]
print(f"Squares (comprehension): {squares_comprehension}")

# Example 2: Filtering elements with a condition
even_numbers = [num for num in numbers if num % 2 == 0]
print(f"Even numbers: {even_numbers}")

# Example 3: List comprehension with nested conditions
filtered_squares = [num ** 2 for num in range(10) if num % 2 == 0 if num > 2]
print(f"Filtered squares (nested conditions): {filtered_squares}") # Squares of even numbers greater than 2: [16, 36, 64]

# Example 4: List comprehension with if-else expression (conditional output)
parity_labels = ["Even" if num % 2 == 0 else "Odd" for num in range(1, 6)]
print(f"Parity labels: {parity_labels}") # Output: ['Odd', 'Even', 'Odd', 'Even', 'Odd']