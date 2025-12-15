"""
EXERCISE 1

Simple Linear Regression: Create a simple linear regression model to predict Final_Exam_Score using only the Study_Hours feature.

    - Split the data (using X_study = data[['Study_Hours']] and y = data['Final_Exam_Score']).
    - Train the model.
    - Print the intercept and coefficient.
    - Make predictions on the test set.
    - Calculate and print the MSE, RMSE, and R-squared.
    - Plot the actual vs. predicted values for this simple model. How does the R-squared compare to the multiple linear regression model?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 200

# Generate synthetic data with realistic relationships
study_hours = np.random.uniform(1, 10, n_samples)
attendance = np.random.uniform(50, 100, n_samples)
previous_grade = np.random.uniform(60, 95, n_samples)

# Final exam score with some realistic relationships and noise
final_exam_score = (
    30 +  # Base score
    3.5 * study_hours +  # Study hours effect
    0.2 * attendance +  # Attendance effect
    0.3 * previous_grade +  # Previous grade effect
    np.random.normal(0, 5, n_samples)  # Random noise
)

# Clip scores to realistic range [0, 100]
final_exam_score = np.clip(final_exam_score, 0, 100)

# Create DataFrame
data = pd.DataFrame({
    'Study_Hours': study_hours,
    'Attendance': attendance,
    'Previous_Grade': previous_grade,
    'Final_Exam_Score': final_exam_score
})

print("="*80)
print("SIMPLE LINEAR REGRESSION: Study Hours → Final Exam Score")
print("="*80)

# Display first few rows of data
print("\nDataset Preview:")
print(data.head(10))
print(f"\nDataset shape: {data.shape}")
print(f"\nBasic statistics:")
print(data[['Study_Hours', 'Final_Exam_Score']].describe())

"""
STEP 1: PREPARE THE DATA
"""
print("\n" + "="*80)
print("STEP 1: DATA PREPARATION")
print("="*80)

# Select features and target
X_study = data[['Study_Hours']]  # Feature (must be 2D for sklearn)
y = data['Final_Exam_Score']      # Target

print(f"\nFeature (X) shape: {X_study.shape}")
print(f"Target (y) shape: {y.shape}")

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_study, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

"""
STEP 2: TRAIN THE MODEL
"""
print("\n" + "="*80)
print("STEP 2: TRAIN THE MODEL")
print("="*80)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

"""
STEP 3: PRINT INTERCEPT AND COEFFICIENT
"""
print("\n" + "="*80)
print("STEP 3: PRINT INTERCEPT AND COEFFICIENT")
print("="*80)

intercept = model.intercept_
coefficient = model.coef_[0]

print(f"\nIntercept (β₀): {intercept:.4f}")
print(f"Coefficient (β₁): {coefficient:.4f}")

print(f"\nRegression Equation:")
print(f"Final_Exam_Score = {intercept:.4f} + {coefficient:.4f} × Study_Hours")

print(f"\nInterpretation:")
print(f"   • Intercept: Expected score with 0 study hours = {intercept:.2f}")
print(f"   • Coefficient: For each additional hour of study, score increases by {coefficient:.2f} points")

"""
STEP 4: MAKE PREDICTIONS
"""
print("\n" + "="*80)
print("STEP 4: PREDICTIONS")
print("="*80)

# Make predictions on test set
y_pred = model.predict(X_test)

# Display sample predictions
print("\nSample Predictions (first 10 test samples):")
comparison_df = pd.DataFrame({
    'Study_Hours': X_test['Study_Hours'].values[:10],
    'Actual_Score': y_test.values[:10],
    'Predicted_Score': y_pred[:10],
    'Error': y_test.values[:10] - y_pred[:10]
})
print(comparison_df.to_string(index=False))

"""
STEP 5: CALCULATE METRICS
"""

print("\n" + "="*80)
print("STEP 5: MODEL EVALUATION METRICS")
print("="*80)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Also calculate for training set to check for overfitting
y_train_pred = model.predict(X_train)
r2_train = r2_score(y_train, y_train_pred)

print("\nTest Set Metrics:")
print(f"   • MSE (Mean Squared Error): {mse:.4f}")
print(f"   • RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"   • R² Score: {r2:.4f}")

print("\nTraining Set Metrics:")
print(f"   • R² Score (Train): {r2_train:.4f}")

print("\nMetric Interpretations:")
print(f"   • RMSE = {rmse:.2f}: On average, predictions are off by ±{rmse:.2f} points")
print(f"   • R² = {r2:.4f}: Model explains {r2*100:.2f}% of variance in exam scores")

if abs(r2_train - r2) < 0.05:
    print(f"   • Model generalization: Good (train-test R² difference < 5%)")
elif abs(r2_train - r2) < 0.10:
    print(f"   • Model generalization: Acceptable (train-test R² difference < 10%)")
else:
    print(f"   • Model generalization: May be overfitting (train-test R² difference > 10%)")

"""
STEP 6: VISUALIZATION
"""

print("\n" + "="*80)
print("STEP 6: VISUALIZATIONS")
print("="*80)

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Simple Linear Regression: Study Hours → Final Exam Score', 
             fontsize=16, fontweight='bold')

# Plot 1: Scatter plot with regression line (Training data)
ax1 = axes[0, 0]
ax1.scatter(X_train, y_train, alpha=0.6, color='steelblue', 
           edgecolors='black', linewidth=0.5, s=60, label='Training data')
ax1.scatter(X_test, y_test, alpha=0.6, color='coral', 
           edgecolors='black', linewidth=0.5, s=60, label='Test data')

# Plot regression line
X_line = np.linspace(X_study.min(), X_study.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
ax1.plot(X_line, y_line, 'r-', linewidth=2.5, label='Regression line', alpha=0.8)

ax1.set_xlabel('Study Hours', fontsize=11, fontweight='bold')
ax1.set_ylabel('Final Exam Score', fontsize=11, fontweight='bold')
ax1.set_title('Scatter Plot with Regression Line', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
        fontsize=11, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Actual vs Predicted (Test set)
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred, alpha=0.6, color='green', 
           edgecolors='black', linewidth=0.5, s=60)

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', 
        linewidth=2, label='Perfect prediction', alpha=0.8)

ax2.set_xlabel('Actual Exam Score', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Exam Score', fontsize=11, fontweight='bold')
ax2.set_title('Actual vs Predicted Values (Test Set)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, f'RMSE = {rmse:.2f}', transform=ax2.transAxes, 
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Plot 3: Residuals plot
ax3 = axes[1, 0]
residuals = y_test - y_pred
ax3.scatter(y_pred, residuals, alpha=0.6, color='purple', 
           edgecolors='black', linewidth=0.5, s=60)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.8)
ax3.set_xlabel('Predicted Exam Score', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
ax3.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Distribution of residuals
ax4 = axes[1, 1]
ax4.hist(residuals, bins=20, color='orange', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, alpha=0.8)
ax4.set_xlabel('Residuals', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.text(0.05, 0.95, f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}', 
        transform=ax4.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('simple_linear_regression_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations created successfully!")
print("  Plot saved as: 'simple_linear_regression_analysis.png'")
plt.show()

"""
COMPARISON: SIMPLE LINEAR REGRESSION VS MULTIPLE LINEAR REGRESSION
"""

print("\n" + "="*80)
print("COMPARISON: SIMPLE vs MULTIPLE LINEAR REGRESSION")
print("="*80)

# Train a multiple linear regression model for comparison
X_multiple = data[['Study_Hours', 'Attendance', 'Previous_Grade']]
X_train_mult, X_test_mult, y_train_mult, y_test_mult = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42
)

model_multiple = LinearRegression()
model_multiple.fit(X_train_mult, y_train_mult)
y_pred_mult = model_multiple.predict(X_test_mult)

r2_multiple = r2_score(y_test_mult, y_pred_mult)
rmse_multiple = np.sqrt(mean_squared_error(y_test_mult, y_pred_mult))

print("\nModel Comparison:")
print(f"\n{'Metric':<25} {'Simple LR':<15} {'Multiple LR':<15} {'Difference':<15}")
print("-" * 70)
print(f"{'R² Score':<25} {r2:<15.4f} {r2_multiple:<15.4f} {r2_multiple - r2:<15.4f}")
print(f"{'RMSE':<25} {rmse:<15.4f} {rmse_multiple:<15.4f} {rmse_multiple - rmse:<15.4f}")
print(f"{'Features Used':<25} {1:<15} {3:<15} {2:<15}")

improvement = ((r2_multiple - r2) / r2) * 100 if r2 > 0 else 0

print(f"\nAnalysis:")
print(f"   • Simple model R² = {r2:.4f} (explains {r2*100:.2f}% of variance)")
print(f"   • Multiple model R² = {r2_multiple:.4f} (explains {r2_multiple*100:.2f}% of variance)")
print(f"   • Improvement: {improvement:.2f}% increase in explained variance")

if r2_multiple > r2:
    print(f"   • Adding Attendance and Previous_Grade improves prediction accuracy")
else:
    print(f"   • Additional features do not significantly improve the model")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
