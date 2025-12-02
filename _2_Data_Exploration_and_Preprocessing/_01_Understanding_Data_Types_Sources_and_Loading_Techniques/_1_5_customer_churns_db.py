import pandas as pd

# For demonstration purposes, we'll use SQLite, which doesn't require a separate server.
# This will create a temporary in-memory database.
import sqlite3

# Create a connection to an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Create a table and insert some customer churn data
conn.execute("""
CREATE TABLE IF NOT EXISTS ChurnCustomers (
    CustomerID INTEGER PRIMARY KEY,
    Gender TEXT,
    Age INTEGER,
    MonthlyCharges REAL,
    Churn TEXT
);
""")
conn.execute("INSERT INTO ChurnCustomers VALUES (1001, 'Male', 34, 50.00, 'No');")
conn.execute("INSERT INTO ChurnCustomers VALUES (1002, 'Female', 56, 80.50, 'Yes');")
conn.execute("INSERT INTO ChurnCustomers VALUES (1003, 'Female', 22, 30.00, 'No');")
conn.commit()

# Load data using pd.read_sql_query
# This executes a SQL query and returns the results as a DataFrame
df_sql_query = pd.read_sql_query("SELECT * FROM ChurnCustomers WHERE Age > 30;", conn)
print("\nDataFrame loaded from SQL Query (Age > 30):")
print(df_sql_query.head())

# Load an entire table using pd.read_sql_table (requires SQLAlchemy for connection string parsing)
# For simplicity, we stick to read_sql_query with a direct connection object in this basic example.
# A full production setup would involve SQLAlchemy for more robust connection management.

# Close the connection
conn.close()