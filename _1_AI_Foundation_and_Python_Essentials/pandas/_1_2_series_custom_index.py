import pandas as pd

temperatures = [25, 28, 22, 19, 30]
cities = ['Tokyo', 'Kathmandu', 'Delhi', 'Sao Paulo', 'London']
city_temps_labeled = pd.Series(temperatures, index=cities)

print("\nSeries with custom string index:")
print(city_temps_labeled)
print("\nAccessing a specific value by label ('Kathmandu'):", city_temps_labeled['Kathmandu'])
print("Accessing a specific value by position (0):", city_temps_labeled[0]) # Positional indexing still works but it is deprecated