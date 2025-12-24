import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Simulate actual and predicted values for a regression task
# Let's imagine predicting the salary of employees (in thousands of USD)
actual_salaries = np.array([50, 65, 70, 80, 95, 110, 120, 55, 75, 100])
predicted_salaries = np.array([52, 63, 72, 78, 98, 105, 125, 58, 70, 95])

print("Actual Salaries:", actual_salaries)
print("Predicted Salaries:", predicted_salaries)

# Calculate MAE
mae = mean_absolute_error(actual_salaries, predicted_salaries)
print(f"\nMean Absolute Error (MAE): {mae:.2f}")

# Calculate MSE
mse = mean_squared_error(actual_salaries, predicted_salaries)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Calculate RMSE (Root Mean Squared Error) for better interpretability of MSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Calculate R-squared
r2 = r2_score(actual_salaries, predicted_salaries)
print(f"R-squared (R2): {r2:.2f}")

# --- Demonstrating sensitivity to outliers ---
# Let's add an outlier to the actual salaries and see how MSE and MAE change

actual_salaries_with_outlier = np.array([50, 65, 70, 80, 95, 110, 120, 55, 75, 100, 200]) # Added 200 as an outlier
predicted_salaries_with_outlier = np.array([52, 63, 72, 78, 98, 105, 125, 58, 70, 95, 110]) # Corresponding prediction (bad one)

print("\n--- Metrics with an Outlier ---")
print("Actual Salaries with Outlier:", actual_salaries_with_outlier)
print("Predicted Salaries with Outlier:", predicted_salaries_with_outlier)

mae_outlier = mean_absolute_error(actual_salaries_with_outlier, predicted_salaries_with_outlier)
print(f"MAE with outlier: {mae_outlier:.2f}")

mse_outlier = mean_squared_error(actual_salaries_with_outlier, predicted_salaries_with_outlier)
print(f"MSE with outlier: {mse_outlier:.2f}")

rmse_outlier = np.sqrt(mse_outlier)
print(f"RMSE with outlier: {rmse_outlier:.2f}")

r2_outlier = r2_score(actual_salaries_with_outlier, predicted_salaries_with_outlier)
print(f"R2 with outlier: {r2_outlier:.2f}")

# Observe that MSE increased significantly more than MAE due to the squared term penalizing the large error (200-110 = 90) heavily.
# The R2 score also dropped significantly, indicating a poorer fit.