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
    