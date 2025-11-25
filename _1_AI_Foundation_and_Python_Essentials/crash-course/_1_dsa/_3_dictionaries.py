# 3. DICTIONARIES
# -> unordered collections of items where each item consists of a key-value pair
# -> mutable meaning you can add, remove, or modify key-value pairs after creation
# -> key-value
#       -> key: Keys must be unique and immutable (like strings, numbers, or tuples),
#       -> value: Values can be of any data type and can be duplicates.
# -> useful for representing structured data where you need to look up information by a specific identifier.

# Creating Dictionaries
# -> defined by enclosing comma-separated key-value pairs within curly braces {}.
# -> Each key-value pair is separated by a colon :.

# An empty dictionary
empty_dict = {}
print(f"Empty dictionary: {empty_dict}")

# A dictionary storing a customer's details
customer = {
    "customer_id": "C001",
    "name": "Kawayama Sato",
    "age": 45,
    "is_active": True,
    "last_purchase_date": "2025-11-11"
}
print(f"Customer dictionary: {customer}")

# A dictionary mapping product IDs to their prices
product_prices = {
    "P101": 29.99,
    "P102": 15.50,
    "P103": 5.99
}
print(f"Product prices: {product_prices}")

#
# Accessing and Modifying Dictionary Elements
# Values are accessed using their corresponding keys.

print("\nAccessing and Modifying Dictionary Elements")
# Accessing values
customer_name = customer["name"]
customer_age = customer["age"]
print(f"Name: {customer_name}, Age: {customer_age}")

# Using .get() for safe access (returns None or a default if key not found)
customer_email = customer.get("email", "Not Provided")
print(f"Customer email: {customer_email}") # Output: Not Provided (key doesn't exist)

# Adding a new key-value pair
customer["email"] = "hoge@example.com"
print(f"Customer dictionary after adding email: {customer}")

# Modifying an existing value
customer["age"] = 46
print(f"Customer dictionary after updating age: {customer}")

# Removing a key-value pair
del customer["last_purchase_date"]
print(f"Customer dictionary after deleting last_purchase_date: {customer}")

# Using .pop() to remove and retrieve a value
is_active_status = customer.pop("is_active")
print(f"Is active status removed: {is_active_status}, Dictionary after pop: {customer}")

#
# Dictionary Methods
# -> Dictionaries come with useful methods for retrieving keys, values, or items (key-value pairs).

print("\nDictionary Methods")

# Get all keys
all_keys = customer.keys()
print(f"All keys: {list(all_keys)}") # Convert to list for easier viewing

# Get all values
all_values = customer.values()
print(f"All values: {list(all_values)}")

# Get all items (key-value pairs as tuples)
all_items = customer.items()
print(f"All items: {list(all_items)}")

# Update a dictionary with another dictionary
new_details = {"phone": "555-1234", "age": 47} # 'age' will be updated
customer.update(new_details)
print(f"Customer after update: {customer}")