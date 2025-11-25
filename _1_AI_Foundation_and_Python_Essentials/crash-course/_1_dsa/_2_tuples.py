# 2. TUPLES
# -> similar to lists in that they are ordered collections of items and can store heterogeneous data.
# -> immutable meaning Once a tuple is created, its contents cannot be changed (elements cannot be added, removed, or modified).

# Creating Tuples
# -> defined by enclosing comma-separated elements within parentheses ().
# -> Parentheses are optional when defining tuples, as long as there's a comma for multiple items.
# -> A single item tuple requires a trailing comma.

# An empty tuple
empty_tuple = ()
print(f"Empty tuple: {empty_tuple}")

# A tuple of coordinates (latitude, longitude)
coordinates = (34.0522, -118.2437)
print(f"Coordinates tuple: {coordinates}")

# A tuple of RGB color values
rgb_color = (255, 0, 128)
print(f"RGB color tuple: {rgb_color}")

# Tuple without parentheses (tuple packing)
sensor_data = 10.5, "temperature", "C"
print(f"Sensor data tuple (packed): {sensor_data}")

# Single item tuple requires a trailing comma
single_item_tuple = (42,)
print(f"Single item tuple: {single_item_tuple}")

#
# Accessing Elements: Indexing and Slicing
# -> Just like lists, tuple elements are accessed using indexing and slicing.

print("\nAccessing Elements: Indexing and Slicing")
user_profile = ("john_doe", 30, "Software Engineer", True)

# Accessing elements
name = user_profile[0]
age = user_profile[1]
print(f"User name: {name}, Age: {age}")

# Slicing a tuple
job_status = user_profile[2:]
print(f"Job and active status: {job_status}")

#
# Immutability
# -> Attempting to modify a tuple will result in a TypeError.
# -> This immutability makes tuples suitable for data that should not change,
#       such as configuration settings, fixed coordinates, or as dictionary keys (which lists cannot be).

print("\nImmutability")

coordinates = (34.5, 35.0)

# Attempting to modify a tuple will raise an error
try:
    coordinates[0] = 34.0
except TypeError as e:
    print(f"Error modifying tuple: {e}")

# Example of when tuples are useful:
# When a function returns multiple values, it often returns them as a tuple.
def get_sensor_reading():
    # In a real scenario, this would read from a sensor
    return 25.5, "Celsius", "OK"

temp, unit, status = get_sensor_reading() # Tuple unpacking
print(f"Temperature: {temp}, Unit: {unit}, Status: {status}")