"""
1. List Manipulation:

    Create a list called sensor_data with the following floating-point values: [22.5, 23.1, 21.9, 24.0, 22.8, 25.5].
    Add a new reading 23.7 to the end of the list.
    Insert a value 20.0 at the beginning of the list.
    Remove the first occurrence of 22.8 from the list.
    Print the modified list.
    Using a list comprehension, create a new list high_readings containing only the values from sensor_data that are greater than 23.0. Print high_readings.
"""

print(f"Exercise 1: List Manipulation")
# Task 1: Create a list called sensor_data with the following floating-point values: [22.5, 23.1, 21.9, 24.0, 22.8, 25.5].
sensor_data = [22.5, 23.1, 21.9, 24.0, 22.8, 25.5]

print(f"Sensor data: {sensor_data}")

# Task 2: Add a new reading 23.7 to the end of the list.
sensor_data.append(23.7)
print(f"Sensor data after adding a new reading 23.7 to the end of the list: {sensor_data}")

# Task 3: Insert a value 20.0 at the beginning of the list.
sensor_data.insert(0, 20.0)
print(f"Sensor data after inserting a value 20.0 at the beginning of the list: {sensor_data}")

# Task 4: Remove the first occurrence of 22.8 from the list.
sensor_data.remove(22.8)
print(f"Sensor data after removing the first occurrence of 22.8 from the list: {sensor_data}")

# Task 5: Print the modified list.
print(f"Sensor data after modification: {sensor_data}")

# Task 6: Using a list comprehension, create a new list high_readings containing only the values from sensor_data that
# are greater than 23.0. Print high_readings.
# high_readings = []

# for value in sensor_data:
#     if value > 23.0:
#         high_readings.append(value)

high_readings = [value for value in sensor_data if value > 23.0]

print(f"High readings: {high_readings}")

print(f"End of Exercise 1: List Manipulation")
print("\n")

"""
2. Tuple Unpacking and Data Representation:

    - Imagine you have GPS coordinates and associated accuracy as a tuple: (40.7128, -74.0060, 5.2). 
        The values represent (latitude, longitude, accuracy_in_meters).
    - Unpack this tuple into three separate variables: latitude, longitude, accuracy.
    - Print each variable with a descriptive label (e.g., "Latitude: 40.7128").
"""

print(f"Exercise 2: Tuple Unpacking and Data Representation")

coordinates = (40.7128, -74.0060, 5.2)

# Unpack this tuple into three separate variables: latitude, longitude, accuracy.
latitude, longitude, accuracy = coordinates

# Print each variable with a descriptive label (e.g., "Latitude: 40.7128").
print(f"Latitude: {latitude}, Longitude: {longitude}, Accuracy: {accuracy}")

print(f"Exercise 3: Tuple Unpacking and Data Representation")
print("\n")

"""
3. Dictionary Operations:

    - Create a dictionary called user_profile with the following key-value pairs:
        "username": "hoge.fuga"
        "email": "hoge.fuga@example.com"
        "subscription_level": "basic"
        "last_activity_days_ago": 7
    - Update the subscription_level to "premium".
    - Add a new key-value pair "signup_date": "2025-01-29".
    - Attempt to retrieve the value for the key "phone_number". If the key doesn't exist, print "Phone number not available." using the .get() method.
    - Print the final user_profile dictionary.
"""

print(f"Exercise 3: Dictionary Operations")

user_profile = {
    "username": "hoge.fuga",
    "email": "hoge.fuga@example.com",
    "subscription_level": "basic",
    "last_activity_days_ago": 7,
}

# Update the subscription_level to "premium".
user_profile['subscription_level'] = "premium"

# Add a new key-value pair "signup_date": "2025-01-29".
user_profile['signup_date'] = "2025-01-29"

# Attempt to retrieve the value for the key "phone_number". If the key doesn't exist, print "Phone number not available." using the .get() method.
phone_number_status = user_profile.get("phone_number", "Phone number not available.")
print(f"Phone number retrieval status: {phone_number_status}")

# Print the final user_profile dictionary.
print(f"User profile: {user_profile}")

print(f"End of Exercise 3: Dictionary Operations")
print("\n")

"""
4. Set Operations and Membership:

    - Create two sets:
        active_users = {"Akita", "Yamaguchi", "Kim", "Taniguchi"}
        premium_members = {"Yamaguchi", "Aaoyagi", "Kim"}
    - Find and print the names of users who are both active_users AND premium_members.
    - Find and print the names of users who are active_users but NOT premium_members.
    - Check if "Naruto" is in active_users and print the result.
"""

print(f"Exercise 4: Set Operations and Membership")

active_users = {"Akita", "Yamaguchi", "Kim", "Taniguchi"}
premium_members = {"Yamaguchi", "Aaoyagi", "Kim"}

# Find and print the names of users who are both active_users AND premium_members.
# Use .intersection() or the '&' operator
all_users = active_users.intersection(premium_members)
# Alternative syntax: users_in_both = active_users & premium_members
print(f"All users: {all_users}")

# Find and print the names of users who are active_users but NOT premium_members.
only_active_users = active_users.difference(premium_members)
# Alternative syntax: only_active_users = active_users - premium_members
print(f"Active users who are NOT premium: {only_active_users}")

# Check if "Naruto" is in active_users and print the result.
is_naruto_exists = "Naruto" in active_users
print(f"Is Naruto in active_users? {is_naruto_exists}")

print(f"End of Exercise 4: Set Operations and Membership")
print("\n")


"""
5. Control Flow with Loops and Conditionals:

    - You have a list of numerical data_points = [15, 8, 22, -3, 30, 0, 19, 10].
    - Iterate through this list using a for loop.
    - Inside the loop, use if-elif-else statements to categorize each number:
        If the number is negative, print "Negative: [number]".
        If the number is 0, print "Zero: [number]".
        If the number is positive and even, print "Positive Even: [number]".
        If the number is positive and odd, print "Positive Odd: [number]".
    - Modify the loop to stop processing and print "Maximum value reached." if a number greater than 25 is encountered, using the break statement.
"""

print(f"Exercise 5: Control Flow with Loops and Conditionals")
#

data_points = [15, 8, 22, -3, 30, 0, 19, 10]

# Iterate through this list using a for loop.
for number in data_points:
    # Modify the loop to stop processing and print "Maximum value reached." if a number greater than 25 is encountered, using the break statement.
    if number > 25:
        print("Maximum value reached.")
        break

    # If the number is negative, print "Negative: [number]".
    if number < 0:
        print(f"Negative: {number}")
    # If the number is 0, print "Zero: [number]".
    elif number == 0:
        print(f"Zero: {number}")
    # If the number is positive and even, print "Positive Even: [number]".
    elif number > 0 and  number % 2 == 0:
        print(f"Positive Even: {number}")
    # If the number is positive and odd, print "Positive Odd: [number]".
    elif number > 0 and number % 2 == 1:
        print(f"Positive Odd: {number}")

print(f"End of Exercise 5: Control Flow with Loops and Conditionals")
print("\n")