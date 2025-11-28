import pandas as pd
import numpy as np

# Example 3: DataFrame from a NumPy array
# Requires explicit column names if desired.
np_data = np.random.rand(4, 3) * 100 # 4 rows, 3 columns of random numbers
columns = ['FeatureA', 'FeatureB', 'FeatureC']
data_from_np = pd.DataFrame(np_data, columns=columns)
print("\nDataFrame from a NumPy array:")
print(data_from_np)