import pandas as pd

# Example 3: Series from a dictionary (keys become the index)
city_temps_dict = {'Tokyo': 25, 'Kathmandu': 28, 'Delhi': 22, 'Sao Paulo': 19, 'London': 30}
city_temps_from_dict = pd.Series(city_temps_dict)

print("\nSeries from a dictionary (keys become the index):")
print(city_temps_from_dict)
print("\nData type of elements:", city_temps_from_dict.dtype)
print("Index of the Series:", city_temps_from_dict.index)