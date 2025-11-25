# 4. SETS
# -> unordered collections of unique items
# -> mutable, meaning you can add or remove elements.
# -> useful for tasks involving membership testing,
#   removing duplicates from a sequence,
#   and performing mathematical set operations like union, intersection, and difference.

#
# Creating Sets
# -> defined by enclosing comma-separated elements within curly braces {}.
# -> An empty set must be created using set(), not {} (which creates an empty dictionary).


# An empty set
empty_set = set()
print(f"Empty set: {empty_set}")

# A set of unique user IDs
user_ids = {"U001", "U005", "U002", "U001", "U003"} # "U001" is a duplicate
print(f"User IDs set (duplicates removed): {user_ids}") # Output will be unique items in arbitrary order

# A set of categories
product_categories = {"Electronics", "Books", "Clothing", "Books"}
print(f"Product categories: {product_categories}")

#
# Modifying Sets and Set Operations

print("\nModifying Sets and Set Operations")

unique_visitors = {"visitor_A", "visitor_B", "visitor_C"}
new_visitors = {"visitor_C", "visitor_D", "visitor_E"}
print(f"Unique visitors: {unique_visitors}")
print(f"New visitors: {new_visitors}")

# Adding elements
unique_visitors.add("visitor_F")
print(f"\nAfter adding visitor_F: {unique_visitors}")

# Removing elements
unique_visitors.remove("visitor_B") # Raises KeyError if item not found
print(f"After removing visitor_B: {unique_visitors}")

# Discarding elements (no error if item not found)
unique_visitors.discard("visitor_Z") # No error
print(f"After discarding visitor_Z: {unique_visitors}")

# Set operations: Union (all unique elements from both sets)
all_visitors = unique_visitors.union(new_visitors)
print(f"All visitors (union): {all_visitors}")

# Set operations: Intersection (common elements)
common_visitors = unique_visitors.intersection(new_visitors)
print(f"Common visitors (intersection): {common_visitors}")

# Set operations: Difference (elements in unique_visitors but not in new_visitors)
distinct_to_first = unique_visitors.difference(new_visitors)
print(f"Visitors unique to first set: {distinct_to_first}")

# Set operations: Symmetric Difference (elements in either set, but not in both)
either_but_not_both = unique_visitors.symmetric_difference(new_visitors)
print(f"Visitors in either but not both: {either_but_not_both}")

# Checking for membership
is_in_set = "visitor_A" in unique_visitors
print(f"Is visitor_A in unique_visitors? {is_in_set}")
