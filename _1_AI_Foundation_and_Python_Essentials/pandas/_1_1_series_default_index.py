import pandas as pd
    
# Example 1: Basic Series from a list
temperatures = [25, 28, 22, 19, 30]
city_temps = pd.Series(temperatures)

print("Series from a list (default index):")
print(city_temps)
print("\nData type of elements:", city_temps.dtype)
print("Index of the Series:", city_temps.index)