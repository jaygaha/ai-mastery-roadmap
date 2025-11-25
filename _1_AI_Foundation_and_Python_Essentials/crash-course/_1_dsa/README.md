# Python Data Structures

Python offers several built-in data structures that provide flexible and efficient ways to store collections of data. 
Each structure has distinct characteristics regarding mutability (whether it can be changed after creation), order 
(if elements maintain their insertion order), and whether duplicate elements are allowed.

## 1. Lists

Lists are ordered, mutable collections in Python.

**Key Features:**
- Mutable: Contents can be changed after creation.
- Ordered: Elements maintain their position.
- Heterogeneous: Can contain items of different data types.

Use Case: Powerful for representing diverse datasets.

### Example

Check this example implementation [_1_lists.py](_1_lists.py)

## 2. Tuples

**Order & Data Types**

Both tuples and lists are ordered collections capable of storing heterogeneous (mixed) data types.

- Mutability
  - Lists: Mutable—elements can be added, removed, or modified after creation.
  - Tuples: Immutable—once created, the contents cannot be changed. No additions, removals, or modifications are allowed.

### Example

Check this example implementation [_2_tuples.py](_2_tuples.py)

## 3. Dictionaries

Unordered collections of key-value pairs.

**Properties**
- Mutable: Add, remove, or modify pairs after creation.
- Keys: Must be unique and immutable (e.g., strings, numbers, tuples).
- Values: Can be any data type; duplicates allowed.

Use Case: Ideal for structured data requiring lookup by identifier.

### Example

Check this example implementation [_3_dictionaries.py](_3_dictionaries.py)

## 4. Sets

Unordered collections of unique items.

**Key Properties**
- Mutable: Elements can be added or removed.
- Uniqueness: Each item appears only once.

**Common Use Cases**
- Membership testing (checking if an item exists).
- Removing duplicates from sequences.
- Mathematical set operations:
  - Union
  - Intersection
  - Difference

### Example

Check this example implementation [_4_sets.py](_4_sets.py)