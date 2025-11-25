# 2. LOOPS
# 1. FOR
#  -> A for loop is used for iterating over a sequence (such as a list, tuple, string, or range).
#  -> It is best used when you know the number of times you want the code to repeat.

print("\n1. for Loops")

#
# Iterating Over Lists and Tuples
print("\nIterating Over Lists and Tuples")
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

product_ids = ["P_001", "P_002", "P_003", "P_004"]
for pid in product_ids:
    print(f"Processing product ID: {pid}")

customer_details = ("Alice", 30, "Premium")
for detail in customer_details:
    print(f"Detail: {detail}")

#
# Iterating Over Strings
print("\nIterating Over Strings")

message = "Python"
for char in message:
    print(f"Character: {char}")

#
# Using the range() function

print("\nrange() function")

# range(stop): Generates numbers from 0 up to stop-1.

print("\nrange(stop)")

# Loop 5 times (0 to 4)
for i in range(5):
    print(i)

#
# range(start, stop): Generates numbers from start up to stop-1.
print("\nrange(start, stop)")

# Loop from 10 to 14
for num in range(10, 15):
    print(f"Number: {num}")

# range(start, stop, step): Generates numbers from start up to stop-1, incrementing by step.
print("\nrange(start, stop, step)")

# Loop from 0 to 9, stepping by 2
for count in range(0, 10, 2):
    print(f"Count (even): {count}")

#
# Iterating Over Dictionaries
# You can iterate over a dictionary's keys, values, or key-value pairs using its .keys(), .values(), and .items() methods respectively.
print("\nIterating Over Dictionaries")

user_settings = {
    "theme": "dark",
    "notifications": True,
    "language": "en"
}

print(user_settings)

# Iterate over keys (default)
print("\nKeys:")
for key in user_settings:
    print(key)

# Iterate over values
print("\nValues:")
for value in user_settings.values():
    print(value)

# Iterate over key-value pairs
print("\nItems:")
for key, value in user_settings.items():
    print(f"{key}: {value}")

#
# enumerate() for Index and Value
# -> When you need both the index and the value while iterating over a sequence, enumerate() is very useful.
print("\nenumerate()")
features = ["age", "income", "education", "marital_status"]

print(f"Features: {features}")
for index, feature_name in enumerate(features):
    print(f"Feature {index + 1}: {feature_name}") # Adding 1 to index for human-readable count


#
# 2. WHILE Loops
# -> while loops execute a block of code repeatedly as long as a specified condition remains True.
# -> They are suitable when the number of iterations is not known in advance and depends on some condition being met.

print("\n2. while Loops")

# Example: Countdown timer
countdown = 5
while countdown > 0:
    print(countdown)
    countdown -= 1 # Decrement countdown by 1
print("Blast off!")

# Example: User input validation
print("\nUser validation")

user_input = ""
while user_input.lower() not in ["yes", "no"]:
    user_input = input("Please enter 'yes' or 'no': ")
    if user_input.lower() not in ["yes", "no"]:
        print("Invalid input. Try again.")
print(f"You entered: {user_input}")

#
# Hypothetical Scenario: Simulating a sensor reading until a threshold is met

import random

current_value = 0.0
target_value = 10.0
iteration = 0

print("\nSimulating sensor readings...")
while current_value < target_value:
    current_value += random.uniform(0.5, 2.0)  # Add a random increment
    iteration += 1
    print(f"Iteration {iteration}: Current value = {current_value:.2f}")
    if iteration > 10:  # Safety break to prevent infinite loops in simulation
        print("Simulation taking too long, stopping.")
        break
print(f"Target value ({target_value:.2f}) reached or simulation stopped after {iteration} iterations.")

#
# Loop Control Statements: break and continue

#
# break Statement
# -> The break statement immediately terminates the current loop (both for and while) and transfers control to the
#    statement immediately following the loop.
print("\nbreak statement")

search_list = [10, 25, 30, 45, 50, 60]
target = 45

print(f"Searching for {target} in: {search_list}")

for number in search_list:
    if number == target:
        print(f"Target {target} found!")
        break # Exit the loop as soon as the target is found
    print(f"Checking number: {number}")

#
# continue Statement
# -> The continue statement skips the rest of the code inside the current loop iteration and moves to the next iteration.
print("\ncontinue statement")

data_points = [1, 0, 5, -2, 8, 0, 12]

print(f"Processing positive data points: {data_points}")
for dp in data_points:
    if dp <= 0:
        print(f"Skipping non-positive value: {dp}")
        continue # Skip to the next iteration if the data point is not positive
    print(f"Processing positive value: {dp}")