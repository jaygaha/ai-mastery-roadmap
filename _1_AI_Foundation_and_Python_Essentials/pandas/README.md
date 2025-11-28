# Practical Python for Data Analysis: Pandas Introduction

Data analysis is a critical setp in any AI or machine learning pipeline. Before we can train sophisticated models or derive meaningful insights, we need to clean and preprocess our data. *NumPy* provides powerful tools for numerical operations on arrays but often falls short when dealing with structured, tabular data resembling spreadsheets or databases. This is where *Pandas* comes in. *Pandas* is a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation library built on top of the `Python` programming language. *Pandas* introduces two new data structures to Python: the `Series` and the `DataFrame`, which are designed specifically to handle labeled data efficiently.Mastering *Pandas* is an indispensable skill for anyone working in AI because it serves as the foundation for loading, cleaning, exploring, and preparing datam thus bridging the gap between raw data and machine learning ready datasets.

## Main Concepts and Principles

Pandas offers intuitive and efficient methods for storing and manipulating structured data. Its two primary data structures, `Series` and `DataFrames`, are essential tools for data analysis in `Python`.

### Pandas Series: The Labeled One-Dimensional Array

A Pandas Series is a one-dimensional, labeled array that can hold any data type, such as integers, strings, floats, and Python objects. It's essentially a NumPy array with an associated axis label called the index. This index makes data access and alignment more intuitive compared to raw NumPy arrays.

Consider a simple list of temperatures for different cities. In a standard Python list, you might store the values `[25, 28, 22]` and remember that the first value corresponds to Tokyo, the second to Kathmandu, and so on. With a Pandas Series, however, you can explicitly label these values.

**Key Characteristics of a Series:**

* **Homogeneous data type**: All elements within a Series typically have the same data type.
* **Mutable**: You can change the values stored in a Series.
* **Size-immutable**: While you can change values, you cannot add or remove elements after creation without creating a new Series (though methods like append or drop return new Series objects).
* **Labeled index**: Each element has an associated label, which can be custom-defined or automatically generated (0-based integer index by default).

**Examples:**

1. Creating a Series from a list:
    ```python
    import pandas as pd
    
    # Example 1: Basic Series from a list
    temperatures = [25, 28, 22, 19, 30]
    city_temps = pd.Series(temperatures)

    print("Series from a list (default index):")
    print(city_temps)
    print("\nData type of elements:", city_temps.dtype)
    print("Index of the Series:", city_temps.index)
    ```
    In this example, *Pandas* automatically assigns a default integer index starting from 0. The output shows both the index and the corresponding values.
2. Creating a Series with a Custom Index:
    ```python
    import pandas as pd

    # Example 2: Series with a custom string index
    temperatures = [25, 28, 22, 19, 30]
    cities = ['Tokyo', 'Kathmandu', 'Delhi', 'Sao Paulo', 'London']
    city_temps_labeled = pd.Series(temperatures, index=cities)

    print("\nSeries with custom string index:")
    print(city_temps_labeled)
    print("\nAccessing a specific value by label ('Kathmandu'):", city_temps_labeled['Kathmandu'])
    print("Accessing a specific value by position (0):", city_temps_labeled[0]) # Positional indexing still works but it is deprecated
    ```
    Here, we explicitly provide city names as the index. This makes accessing data much more readable and intuitive. You can access values using both their labels and their integer positions.
3. Creating a Series from a Dictionary:
    ```python
    import pandas as pd

    # Example 3: Series from a dictionary (keys become the index)
    city_temps_dict = {'Tokyo': 25, 'Kathmandu': 28, 'Delhi': 22, 'Sao Paulo': 19, 'London': 30}
    city_temps_from_dict = pd.Series(city_temps_dict)

    print("\nSeries from a dictionary (keys become the index):")
    print(city_temps_from_dict)
    print("\nData type of elements:", city_temps_from_dict.dtype)
    print("Index of the Series:", city_temps_from_dict.index)
    ```
    When creating a Series from a dictionary, the dictionary's keys automatically become the index of the Series, and the dictionary's values become the Seris Data. This is a very convenient way to create Series objects.
4. Series Operations (similar to NumPy):
    ```python
    import pandas as pd

    # Example 4: Basic operations on Series
    s1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
    s2 = pd.Series([4, 5, 6], index=['a', 'b', 'd']) # Note: different index for s2

    print("\nSeries s1:\n", s1)
    print("\nSeries s2:\n", s2)

    # Addition - aligned by index
    print("\nAddition (s1 + s2) - demonstrates index alignment:")
    print(s1 + s2) # 'c' + 'd' results in NaN (Not a Number) because they don't align

    # Scalar multiplication
    print("\nScalar multiplication (s1 * 2):\n", s1 * 2)

    # Filtering
    print("\nFiltering s1 (values > 15):\n", s1[s1 > 15])
    ```
    Operations on Series objects are automatically aligned based on their indices. If an index label doesn't exist in both series during an operation, the result for that label will be `NaN` (not a number), which Pandas uses to represent missing or undefined data. This index alignment is a powerful feature that sets Pandas apart from raw NumPy arrays.

### Pandas DataFrame: The Tabular Data Structure

A *Pandas* DataFrame is a two-dimensional, mutable, potentially heterogeneous, tabular data structure with labeled axes (rows and columns). You can think of it as spreadsheet or a SQL table. The DataFrame is the most commonly used *Pandas* object and is designed to handle the type of structured data encountered in most data analysis scenarios.

A DataFrame is essentiallly a collection of Series objects that share the same index. Each Series represents a column of data, and the index represents the row labels.

**Key Characteristics of a DataFrame:**

* **Two-dimensional**: Data is stored in a tabular format with rows and columns.
* **Mutable**: You can add, remove, or modify rows and columns.
* **Heterogeneous**: Different columns can have different data types (e.g., one column might be strings, another integers, another floats)..
* **Labeled axes**: Both rows (index) and columns have labels, allowing for flexible data access.

**Examples:**

1. Creating a DataFrame from a Dictionary of Lists:
    ```python
    import pandas as pd

    # Example 1: DataFrame from a dictionary where values are lists
    # Each key becomes a column name, and each list becomes the column's data.
    data = {
        'City': ['Tokyo', 'Kathmandu', 'Delhi', 'Sao Paulo', 'London'],
        'Temperature': [25, 28, 22, 19, 30],
        'Humidity': [60, 65, 70, 75, 55],
        'Precipitation': [5.2, 2.1, 0.5, 10.3, 0.0]
    }

    weather_df = pd.DataFrame(data)

    print("DataFrame from a dictionary of lists:")
    print(weather_df)
    ```
    This is one of the most common ways to create a data frame. The dictionary keys become the column headers and the lists become the column data. Pandas automatically assigns an integer index to the rows by default.
2. Creating a DataFrame from a List of Dictionaries:
    ```python
    import pandas as pd

    # Example 2: DataFrame from a list of dictionaries
    # Each dictionary in the list becomes a row. Keys become column names.
    employee_data = [
        { 'Name': 'Hoge', 'Age': 30, 'Department': 'HR', 'Salary': 50000},
        { 'Name': 'Fuga', 'Age': 25, 'Department': 'IT', 'Salary': 60000},
        { 'Name': 'Moge', 'Age': 35, 'Department': 'Marketing', 'Salary': 55000}
    ]

    employees_df = pd.DataFrame(employee_data)
    print("\nDataFrame from a list of dictionaries:")
    print(employees_df)
    ```
    In this structure, each dictionary represents a row, and its keys represent the names of the columns. This is useful when your data is structured as records.
3. Creating a DataFrame from a NumPy Array:
    ```python
    import pandas as pd
    import numpy as np

    # Example 3: DataFrame from a NumPy array
    # Requires explicit column names if desired.
    np_data = np.random.rand(4, 3) * 100 # 4 rows, 3 columns of random numbers
    columns = ['FeatureA', 'FeatureB', 'FeatureC']
    data_from_np = pd.DataFrame(np_data, columns=columns)
    print("\nDataFrame from a NumPy array:")
    print(data_from_np)
    ```
    When creating a DataFrame from a NumPy array, you usually provide the array first and then, if you want, you can specify the column names and an index. If you don't specify column names, Pandas will assign default integer column names (0, 1, 2, etc.).


### Practical Examples and Demonstrations

Now that we understand the basics of data structures, let's explore the practical operations that make Pandas powerful for data analysis. For this lesson, we will use a hypothetical dataset about customer purchases from a small online electronics store.

**Loading Data from CSV**

One of the most common tasks is to load data from external files, especially CSV (Comma Separated Values) files. Pandas provides the `read_csv()` function for this purpose.

First, let's simulate a CSV file.
```python
import pandas as pd
import io # Used to simulate a file from a string

# Simulate a CSV file content
csv_data = """CustomerID,ProductName,Category,Price,Quantity,PurchaseDate
1001,Laptop A,Electronics,1200.00,1,2023-01-15
1002,Smartphone X,Electronics,800.00,2,2023-01-16
1001,Mouse,Accessories,25.50,1,2023-01-17
1003,Keyboard,Accessories,75.00,1,2023-01-18
1002,Headphones,Electronics,150.00,1,2023-01-19
1004,Webcam,Accessories,50.00,1,2023-01-20
1003,Monitor,Electronics,300.00,1,2023-01-21
1001,SSD Drive,Components,80.00,2,2023-01-22
"""

# Use io.StringIO to treat the string as a file
df_purchases = pd.read_csv(io.StringIO(csv_data))

print("DataFrame loaded from CSV:\n")
print(df_purchases)
print("\nType of df_purchases:", type(df_purchases))
```
The `read_csv()` function is highly versatile, with many parameters to handle different delimiters, headers, missing values, and more. For now, a simple load is sufficient.

#### **Basic Data Inspection**

Once data is loaded, the first step is usually to get a quick overview of its structure and content.

1. `.head()` and `.tail()`: View the first and last rows.
```python
# Display the first 5 rows (default)
print("\nFirst 3 rows of the DataFrame:")
print(df_purchases.head(3)) # You can specify the number of rows

# Display the last 2 rows
print("\nLast 2 rows of the DataFrame:")
print(df_purchases.tail(2))
```
These methods are invaluable for quickly checking if the data loaded correctly and understanding its layout.

2. `.info()`: Get a concise summary of the DataFrame.
```python
print("\nDataFrame Info:")
df_purchases.info()
```
`info()` provides:
- The class of the object (`pandas.core.frame.DataFrame`).
- The number of entries (rows) and columns.
- A list of all columns, their count of non-null values, and their data type (`Dtype`).
- Memory usage. This is extremely useful for identifying missing values (where non-null counts are less than the total entries) and checking data types.

3. `.shape` and `.columns`: Get dimensions and column names.
```python
print("\nShape of the DataFrame (rows, columns):", df_purchases.shape)
print("Column names:", df_purchases.columns)
```
- `.shape` returns a tuple `(rows, columns)`. 
- `.columns` returns an Index object containing the column labels.

4. `.describe()`: Generate descriptive statistics.
```python
print("\nDescriptive statistics for numerical columns:")
print(df_purchases.describe())
```
For numerical columns, `describe()` provides statistics like count, mean, standard deviation, min, max, and quartiles. It's a quick way to get a sense of the distribution of your numerical data.

#### **Selecting Data (Indexing and Slicing)**

Accessing specific subsets of your data is fundamental. Pandas offers several powerful ways to do this.

1. **Selecting a Single Column**: Returns a Series.
    ```python
    # Select 'ProductName' column using dictionary-like syntax
    product_names = df_purchases['ProductName']
    print("\n'ProductName' column (Series):\n", product_names.head())
    print("Type of product_names:", type(product_names))

    # Alternatively, using dot notation (if column name is a valid Python identifier)
    # This is concise but can be ambiguous if column name clashes with DataFrame methods.
    prices = df_purchases.Price
    print("\n'Price' column (Series) using dot notation:\n", prices.head())
    ```

2. **Selecting Multiple Columns**: Returns a DataFrame.
    ```python
    # Select 'ProductName' and 'Price' columns
    products_and_prices = df_purchases[['ProductName', 'Price']]
    print("\n'ProductName' and 'Price' columns (DataFrame):\n", products_and_prices.head())
    print("Type of products_and_prices:", type(products_and_prices))
    ```
    Notice the double square brackets `[[]]`. The outer brackets indicate a selection, and the inner brackets contain a list of column names.

3. **Selecting Rows by Position with `.iloc`**: (Integer Location)
    ```python
    # Select the first row (index 0)
    first_row = df_purchases.iloc[0]
    print("\nFirst row using .iloc[0]:\n", first_row)
    print("Type of first_row:", type(first_row)) # A Series

    # Select rows from index 1 up to (but not including) 4
    rows_1_to_3 = df_purchases.iloc[1:4]
    print("\nRows from index 1 to 3 using .iloc[1:4]:\n", rows_1_to_3)

    # Select specific non-contiguous rows and columns (e.g., rows 0, 2, and columns 0, 2)
    specific_selection = df_purchases.iloc[[0, 2], [0, 2]]
    print("\nSpecific rows (0, 2) and columns (0, 2) using .iloc[[0, 2], [0, 2]]:\n", specific_selection)
    ```
    `.iloc` is strictly integer-based. The slicing behavior is similar to Python list slicing: the start index is inclusive, and the stop index is exclusive.

4. **Selecting Rows by Label with `.loc`**: (Label Location)
    
    For our current DataFrame, the row labels are the default integer index (0, 1, 2...). `loc` would behave similarly to `iloc` for row selection if you're using default integer labels. However, `loc` shines when you have custom, non-integer row labels (which we don't have here, but it's important to understand).
    ```python
    # Select row with index label 0 (same as iloc[0] for default integer index)
    row_by_label_0 = df_purchases.loc[0]
    print("\nRow with index label 0 using .loc[0]:\n", row_by_label_0)

    # Select rows from label 1 to label 3 (inclusive with .loc)
    # This is a key difference: .loc slicing is inclusive of the stop label
    rows_label_1_to_3 = df_purchases.loc[1:3]
    print("\nRows from label 1 to 3 (inclusive) using .loc[1:3]:\n", rows_label_1_to_3)

    # Select specific rows and columns by labels
    # e.g., rows with label 0 and 2, columns 'ProductName' and 'Price'
    specific_label_selection = df_purchases.loc[[0, 2], ['ProductName', 'Price']]
    print("\nSpecific rows (0, 2) and columns ('ProductName', 'Price') using .loc:\n", specific_label_selection)
    ```
    Remember: `.loc` works with labels, `.iloc` works with integer positions. When slicing with `.loc`, both the start and stop labels are inclusive.

#### **Filtering Data (Boolean Indexing)**

Filtering rows based on conditions is a core data analysis task. Pandas allows you to use boolean (True/False) Series to select rows.
```python
# Select all purchases where the 'Quantity' is greater than 1
high_quantity_purchases = df_purchases[df_purchases['Quantity'] > 1]
print("\nPurchases with Quantity > 1:\n", high_quantity_purchases)

# Select all purchases in the 'Electronics' category
electronics_purchases = df_purchases[df_purchases['Category'] == 'Electronics']
print("\nElectronics purchases:\n", electronics_purchases)

# Combine multiple conditions using logical operators (& for AND, | for OR)
# Purchases in 'Electronics' category AND Price > 500
expensive_electronics = df_purchases[
    (df_purchases['Category'] == 'Electronics') &
    (df_purchases['Price'] > 500)
]
print("\nExpensive Electronics purchases:\n", expensive_electronics)

# Purchases by CustomerID 1001 OR in 'Accessories' category
customer_1001_or_accessories = df_purchases[
    (df_purchases['CustomerID'] == 1001) |
    (df_purchases['Category'] == 'Accessories')
]
print("\nPurchases by Customer 1001 OR in Accessories:\n", customer_1001_or_accessories)

# Using .isin() for multiple discrete values
# Select products that are either 'Laptop A' or 'Monitor'
specific_products = df_purchases[df_purchases['ProductName'].isin(['Laptop A', 'Monitor'])]
print("\nSpecific products ('Laptop A', 'Monitor'):\n", specific_products)
```
- Boolean indexing is incredibly powerful and flexible. 
- Note the use of parentheses around each condition when combining them; this is necessary due to operator precedence in Python.

#### **Adding, Modifying, and Deleting Columns**

DataFrames are `mutable`, meaning you can easily change their structure.

1. Adding a New Column:
    ```python
    # Add a 'TotalAmount' column (Price * Quantity)
    df_purchases['TotalAmount'] = df_purchases['Price'] * df_purchases['Quantity']
    print("\nDataFrame with 'TotalAmount' column added:\n", df_purchases)

    # Add a 'DiscountedPrice' column (e.g., 10% discount)
    df_purchases['DiscountedPrice'] = df_purchases['Price'] * 0.90
    print("\nDataFrame with 'DiscountedPrice' column added:\n", df_purchases)
    ```
    Adding a new column is as simple as assigning a Series (or a scalar value, which will be broadcast) to a new column name.

2. Modifying an Existing Column:
    ```python
    # Increase the price of all accessories by 5%
    # We first select the rows where Category is 'Accessories'
    # Then we select the 'Price' column for those rows and update it.
    df_purchases.loc[df_purchases['Category'] == 'Accessories', 'Price'] *= 1.05
    print("\nDataFrame after increasing 'Accessories' prices by 5%:\n", df_purchases)
    ```
    Using `.loc` for modification is crucial to avoid "SettingWithCopyWarning" and ensure the changes are made directly on the original DataFrame.

3. Deleting Columns:
    ```python
    # Delete the 'DiscountedPrice' column using .drop()
    # axis=1 specifies to drop a column, inplace=True modifies the DataFrame directly
    df_purchases.drop('DiscountedPrice', axis=1, inplace=True)
    print("\nDataFrame after deleting 'DiscountedPrice' column:\n", df_purchases)

    # Alternatively, using 'del' keyword (less common for multiple columns)
    del df_purchases['TotalAmount']
    print("\nDataFrame after deleting 'TotalAmount' column with 'del':\n", df_purchases)
    ```
    `drop()` is the preferred method as it's more flexible (can drop multiple columns or rows, and return a new DataFrame without modification if   `inplace=False`). `axis=1` is essential for columns; `axis=0` is for rows (index). `inplace=True` directly modifies the DataFrame; if `False` (default), it returns a new DataFrame with the column(s) dropped.

#### **Handling Missing Values (Basic Introduction)**

Missing data is a common challenge. Pandas provides tools to identify and handle it. We'll introduce a missing value for demonstration.
```python
# Introduce a NaN value for demonstration
df_purchases.loc[3, 'Price'] = np.nan # Make price missing for the 'Keyboard' purchase

print("\nDataFrame with a missing 'Price' value:\n", df_purchases)

# Check for missing values using .isnull()
print("\nDataFrame indicating missing values (.isnull()):\n", df_purchases.isnull())

# Count missing values per column
print("\nNumber of missing values per column:\n", df_purchases.isnull().sum())

# Drop rows with any missing values
# Caution: This can remove a lot of data if many rows have missing values.
df_no_nan_rows = df_purchases.dropna()
print("\nDataFrame after dropping rows with any NaN values:\n", df_no_nan_rows)

# Fill missing values with a specific value (e.g., 0)
df_filled_zero = df_purchases.fillna(0)
print("\nDataFrame after filling NaN with 0:\n", df_filled_zero)

# Fill missing numerical values with the mean of the column
# We only fill the 'Price' column's NaN values
mean_price = df_purchases['Price'].mean()
df_filled_mean = df_purchases.fillna({'Price': mean_price})
print(f"\nDataFrame after filling 'Price' NaN with mean ({mean_price:.2f}):\n", df_filled_mean)
```
`isnull()` returns a boolean DataFrame of the same shape, indicating True where values are missing. `isnull().sum()` then sums these True values (which are treated as 1) for each column. `dropna()` removes rows (or columns, if `axis=1`) containing any missing values. `fillna()` replaces missing values with a specified value or method (like mean, median, forward-fill, back-fill). More advanced missing value imputation will be covered in later lessons.

#### **Basic Aggregations**

Aggregations allow you to summarize data, typically by grouping it based on certain criteria.
```python
# Calculate the total sales (sum of 'TotalAmount')
total_sales = df_purchases['TotalAmount'].sum()
print(f"\nTotal sales across all purchases: ${total_sales:.2f}")

# Calculate the average price of all products
average_price = df_purchases['Price'].mean()
print(f"Average product price: ${average_price:.2f}")

# Calculate the maximum quantity purchased in a single transaction
max_quantity = df_purchases['Quantity'].max()
print(f"Maximum quantity in a single purchase: {max_quantity}")

# Count unique categories
unique_categories = df_purchases['Category'].nunique()
print(f"Number of unique product categories: {unique_categories}")

# Value counts for 'Category'
category_counts = df_purchases['Category'].value_counts()
print("\nValue counts for 'Category':\n", category_counts)

# Group by 'Category' and calculate the sum of 'TotalAmount' for each category
sales_by_category = df_purchases.groupby('Category')['TotalAmount'].sum()
print("\nTotal sales by Category:\n", sales_by_category)

# Group by 'CustomerID' and calculate the total number of items purchased
items_by_customer = df_purchases.groupby('CustomerID')['Quantity'].sum()
print("\nTotal items purchased by CustomerID:\n", items_by_customer)

# Group by multiple columns and apply multiple aggregations
# For each CustomerID and Category, find the total quantity and average price
customer_category_summary = df_purchases.groupby(['CustomerID', 'Category']).agg(
    TotalQuantity=('Quantity', 'sum'),
    AveragePrice=('Price', 'mean')
)
print("\nCustomer and Category Summary:\n", customer_category_summary)
```
The `groupby()` method is exceptionally powerful. It allows you to split the DataFrame into groups based on one or more columns, apply a function to each group (e.g., sum, mean, count, min, max), and then combine the results into a single DataFrame. This "split-apply-combine" pattern is fundamental to many data analysis tasks.

## Exercises

- [_4_exercise.py](_4_exercise.py)

## Conclusion

In this lesson, we embarked on our journey with Pandas, an essential library for data analysis in Python. We introduced its fundamental data structures: the one-dimensional `Series` and the two-dimensional `DataFrame`, which serve as the cornerstone for handling tabular data. We explored how to create these structures from various data sources—such as lists, dictionaries, and NumPy arrays—and understood the crucial concept of labeled indexing, which distinguishes Pandas from raw numerical arrays.

We then progressed to practical demonstrations, covering key operations such as loading data from CSV files, performing basic data inspection using methods like `head()`, `info()`, and `describe()`, and mastering various techniques for selecting and filtering data using `[]`, `.loc`, `.iloc`, and boolean indexing. We also learned how to dynamically modify DataFrames—adding, modifying, and deleting columns—and received a basic introduction to identifying and handling missing values. Finally, we introduced basic aggregation techniques, such as `groupby()`, to summarize and derive insights from our data.

Pandas is not just a tool—it is a paradigm for interacting with structured data. Its intuitive API, combined with its powerful features like index alignment and efficient operations, make it an indispensable asset in any data science and AI workflow. As you progress through this course, particularly into the data exploration, preprocessing, and model building phases, you will find Pandas to be your most reliable companion. The skills you've gained today are foundational for the upcoming lessons, where you'll apply these techniques to a real-world customer churn prediction case study and prepare data for machine learning models.
