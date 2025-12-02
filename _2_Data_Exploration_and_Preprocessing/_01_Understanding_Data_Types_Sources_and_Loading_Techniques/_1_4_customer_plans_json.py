import pandas as pd

# Create a dummy JSON file
json_data = """
[
    {"CustomerID": 1001, "Plan": "Basic", "Status": "Active"},
    {"CustomerID": 1002, "Plan": "Premium", "Status": "Churned"},
    {"CustomerID": 1003, "Plan": "Basic", "Status": "Active"}
]
"""
with open('_1_4_customer_plans.json', 'w') as f:
    f.write(json_data)

# Load JSON file (list of records)
df_json_plans = pd.read_json('_1_4_customer_plans.json')
print("\nDataFrame loaded from JSON (list of records):")
print(df_json_plans.head())

# Another JSON structure (record-oriented)
json_data_record_oriented = """
{"CustomerID": {"0": 1001, "1": 1002, "2": 1003},
 "Plan": {"0": "Basic", "1": "Premium", "2": "Basic"},
 "Status": {"0": "Active", "1": "Churned", "2": "Active"}}
"""
with open('_1_4_customer_plans_record.json', 'w') as f:
    f.write(json_data_record_oriented)

# Load record-oriented JSON
df_json_record = pd.read_json('_1_4_customer_plans_record.json', orient='columns')
print("\nDataFrame loaded from JSON (record-oriented):")
print(df_json_record.head())