"""
Exercise 2: Customer Churn Prediction (Regression Component)

Problem:
While churn prediction is usually a classification problem (Yes/No), let's imagine we want to predict
Customer Lifetime Value (CLV). This is a regression problem because CLV is a continuous dollar amount.

Scenario:
We have 5 customers. We've predicted their future value, and now we wait to see what they actually spend.
We want to evaluate how good our predictions were using MAE, MSE, RMSE, and R2.

Data:
    Actual CLV:    [150, 300, 80, 450, 200]
    Predicted CLV: [160, 280, 90, 420, 210]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Setup the Data
actual_clv    = np.array([150, 300, 80, 450, 200])
predicted_clv = np.array([160, 280, 90, 420, 210])
customers     = ['C1', 'C2', 'C3', 'C4', 'C5']

# Create a DataFrame to see the errors clearly
df = pd.DataFrame({
    'Customer': customers,
    'Actual': actual_clv,
    'Predicted': predicted_clv,
    'Error': actual_clv - predicted_clv,
    'Abs_Error': np.abs(actual_clv - predicted_clv),
    'Squared_Error': (actual_clv - predicted_clv) ** 2
})

print("\n--- Detailed Prediction Analysis ---")
print(df)


# 2. Calculate Metrics using Scikit-Learn

# MAE (Mean Absolute Error)
mae = mean_absolute_error(actual_clv, predicted_clv)

# MSE (Mean Squared Error)
mse = mean_squared_error(actual_clv, predicted_clv)

# RMSE (Root Mean Squared Error) - The square root of MSE
rmse = np.sqrt(mse)

# R-Squared (R2)
r2 = r2_score(actual_clv, predicted_clv)


# 3. Print the Results with Business Interpretation

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)

print(f"1. MAE:  ${mae:.2f}")
print(f"   Interpretation: On average, our predictions are off by ±${mae:.0f}.")
print(f"   Good for: Reporting to non-technical stakeholders.")

print(f"\n2. MSE:  {mse:.2f}")
print(f"   Interpretation: Hard to interpret directly because the units are squared ($²).")
print(f"   Good for: Optimization algorithms, but bad for reporting.")

print(f"\n3. RMSE: ${rmse:.2f}")
print(f"   Interpretation: Similar to MAE, but penalizes large errors more heavily.")
print(f"   Notice it is slightly higher than MAE (${rmse:.2f} > ${mae:.2f})")
print(f"   because of the larger error for C4 (Error=30).")

print(f"\n4. R2 Score: {r2:.4f}")
print(f"   Interpretation: Our model explains {r2*100:.1f}% of the variation in customer value.")
print(f"   Verdict: This is an excellent score (closer to 1.0 is better).")


# 4. Visualizing the Results (Beginner Friendly Plots)

plt.figure(figsize=(15, 5))

# Plot 1: Actual vs Predicted Comparison
plt.subplot(1, 2, 1)
x = np.arange(len(customers))
width = 0.35

plt.bar(x - width/2, actual_clv, width, label='Actual', color='skyblue')
plt.bar(x + width/2, predicted_clv, width, label='Predicted', color='orange')

plt.xlabel('Customers')
plt.ylabel('CLV ($)')
plt.title('Actual vs Predicted CLV')
plt.xticks(x, customers)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Plot 2: How big are the errors? (Absolute Error)
plt.subplot(1, 2, 2)
plt.bar(customers, df['Abs_Error'], color='salmon')
plt.axhline(y=mae, color='red', linestyle='--', label=f'Average Error (MAE): ${mae:.0f}')

plt.xlabel('Customers')
plt.ylabel('Absolute Error ($)')
plt.title('Prediction Errors per Customer')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()