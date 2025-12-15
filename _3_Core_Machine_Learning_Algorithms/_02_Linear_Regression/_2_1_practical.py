import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42) # for reproducibility

# Feature 1: Study Hours
X_study_hours = np.random.rand(100, 1) * 10 + 2 # 2 to 12 hours
# Feature 2: Previous Exam Score (correlated with study hours)
X_prev_score = X_study_hours * 5 + np.random.randn(100, 1) * 10 + 50 # 50 to 120
# Feature 3: Aptitude Score (less correlated)
X_aptitude = np.random.rand(100, 1) * 50 + 50 # 50 to 100

# Target: Final Exam Score
# Final_Score = 30 + 5*Study_Hours + 0.5*Previous_Exam_Score + 0.2*Aptitude_Score + noise
y_final_score = (30 + 5 * X_study_hours + 0.5 * X_prev_score + 0.2 * X_aptitude +
                 np.random.randn(100, 1) * 15).flatten() # Added some noise and flattened

# Combine features into a DataFrame
data = pd.DataFrame({
    'Study_Hours': X_study_hours.flatten(),
    'Previous_Exam_Score': X_prev_score.flatten(),
    'Aptitude_Score': X_aptitude.flatten(),
    'Final_Exam_Score': y_final_score
})

print("First 5 rows of the synthetic dataset:")
print(data.head())

"""
Data Splitting
"""

print("\nData Splitting:")

# Define features (X) and target (y)
X = data[['Study_Hours', 'Previous_Exam_Score', 'Aptitude_Score']]
y = data['Final_Exam_Score']

# Split data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing
# random_state for reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

"""
Model Training
"""

print("\nModel Training:")

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

print("\nModel training complete.")

"""
Inspecting Model Coefficients
"""

print("\nInspecting Model Coefficients:")

# The intercept (Beta_0)
print(f"\nIntercept (Beta_0): {model.intercept_:.2f}")

# The coefficients for each feature (Beta_1, Beta_2, ..., Beta_n)
print("Coefficients (Beta_n for each feature):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"- {feature}: {coef:.2f}")

"""
Making Predictions
"""

print("\nMaking Predictions:")

# Make predictions on the test set
y_pred = model.predict(X_test)

print("\nFirst 5 actual test scores:", y_test.head().tolist())
print("First 5 predicted test scores:", y_pred[:5].tolist())

"""
Model Evaluation
"""

print("\nModel Evaluation:")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.2f}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2:.2f}")

"""
Visualizing the Results
"""

print("\nVisualizing the Results:")

# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Diagonal line
plt.xlabel("Actual Final Exam Score")
plt.ylabel("Predicted Final Exam Score")
plt.title("Actual vs. Predicted Final Exam Scores")
plt.grid(True)
plt.show()

# Plotting residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel("Predicted Final Exam Score")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

