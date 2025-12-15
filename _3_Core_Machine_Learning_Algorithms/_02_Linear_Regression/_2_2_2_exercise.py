"""
EXERCISE 2

Investigating an Assumption Violation (Hypothetical): Assume you found that the relationship between Study_Hours and Final_Exam_Score was quadratic 
(e.g., Score = a + b*Hours + c*Hours^2).

- Create a new feature called Study_Hours_Squared (data['Study_Hours_Squared'] = data['Study_Hours']**2).
- Now, train a multiple linear regression model using Study_Hours, Study_Hours_Squared, Previous_Exam_Score, and Aptitude_Score as features. (This is still a form of linear 
    regression because it's linear in the coefficients).
- Compare the R-squared of this new model with the original multiple linear regression model. What does this suggest about the importance of the linearity assumption? (You won't 
    necessarily see a huge improvement on our already linear synthetic data, but the exercise is about the concept).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

# Set style for better visualizations
sns.set_style("whitegrid")

print("*"*80)
print("INVESTIGATING NON-LINEAR RELATIONSHIPS WITH POLYNOMIAL FEATURES")
print("*"*80)

"""
STEP 1: Create dataset with NON-LINEAR (quadratic) relationship
"""

print("\n" + "="*80)
print("STEP 1: DATA GENERATION")
print("="*80)

np.random.seed(42)
n_samples = 250

# Generate features
study_hours = np.random.uniform(1, 10, n_samples)
previous_exam_score = np.random.uniform(60, 95, n_samples)
aptitude_score = np.random.uniform(50, 100, n_samples)


# IMPORTANT: Create a QUADRATIC relationship for Study_Hours
# This simulates a realistic scenario: studying helps up to a point, 
# but extreme hours might show diminishing returns or even negative effects (burnout)
final_exam_score = (
    25 +  # Base score
    8 * study_hours +  # Linear effect of study hours
    -0.5 * (study_hours ** 2) +  # QUADRATIC effect (diminishing returns)
    0.3 * previous_exam_score +  # Previous performance
    0.15 * aptitude_score +  # Aptitude effect
    np.random.normal(0, 4, n_samples)  # Random noise
)

# Clip to realistic range
final_exam_score = np.clip(final_exam_score, 0, 100)

# Create DataFrame
data = pd.DataFrame({
    'Study_Hours': study_hours,
    'Previous_Exam_Score': previous_exam_score,
    'Aptitude_Score': aptitude_score,
    'Final_Exam_Score': final_exam_score
})

print("\nDataset Created with QUADRATIC relationship:")
print(f"Score = 25 + 8×Hours - 0.5×Hours² + 0.3×Previous + 0.15×Aptitude + noise")
print("\nDataset Preview:")
print(data.head(10))
print(f"\nDataset shape: {data.shape}")

"""
STEP 2: Create the polynomial feature (Study_Hours_Squared)
"""

print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING - Creating Polynomial Feature")
print("="*80)

# Create the squared feature
data['Study_Hours_Squared'] = data['Study_Hours'] ** 2

print("\n✓ New feature 'Study_Hours_Squared' created!")
print("\nSample of new feature:")
print(data[['Study_Hours', 'Study_Hours_Squared']].head(10))

# Visualize the quadratic relationship
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Visualizing the Quadratic Relationship', fontsize=14, fontweight='bold')

# Plot 1: Scatter plot showing quadratic pattern
ax1 = axes[0]
scatter = ax1.scatter(data['Study_Hours'], data['Final_Exam_Score'], 
                     c=data['Study_Hours'], cmap='viridis', 
                     alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
ax1.set_xlabel('Study Hours', fontsize=11, fontweight='bold')
ax1.set_ylabel('Final Exam Score', fontsize=11, fontweight='bold')
ax1.set_title('Raw Data: Notice the Curved Pattern', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Study Hours')

# Fit a simple polynomial curve for visualization
z = np.polyfit(data['Study_Hours'], data['Final_Exam_Score'], 2)
p = np.poly1d(z)
x_smooth = np.linspace(data['Study_Hours'].min(), data['Study_Hours'].max(), 100)
ax1.plot(x_smooth, p(x_smooth), "r--", linewidth=2.5, alpha=0.8, label='Quadratic trend')
ax1.legend()

# Plot 2: Residuals if we fit only linear model (to show inadequacy)
ax2 = axes[1]
model_linear_simple = LinearRegression()
X_simple = data[['Study_Hours']]
model_linear_simple.fit(X_simple, data['Final_Exam_Score'])
pred_simple = model_linear_simple.predict(X_simple)
residuals_simple = data['Final_Exam_Score'] - pred_simple

ax2.scatter(data['Study_Hours'], residuals_simple, alpha=0.6, 
           color='coral', edgecolors='black', linewidth=0.5, s=50)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Study Hours', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuals (Linear Model Only)', fontsize=11, fontweight='bold')
ax2.set_title('Linear Model Residuals Show Pattern (Bad!)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, 'Pattern in residuals indicates\nnon-linear relationship!', 
        transform=ax2.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()

"""
STEP 3: Train ORIGINAL Multiple Linear Regression (without polynomial term)
"""

print("\n" + "="*80)
print("STEP 3: MODEL 1 - Original Multiple Linear Regression (No Polynomial)")
print("="*80)

# Prepare features (WITHOUT Study_Hours_Squared)
X_original = data[['Study_Hours', 'Previous_Exam_Score', 'Aptitude_Score']]
y = data['Final_Exam_Score']

# Split the data
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_original, y, test_size=0.2, random_state=42
)

# Train the model
model_original = LinearRegression()
model_original.fit(X_train_orig, y_train)

# Make predictions
y_pred_orig = model_original.predict(X_test_orig)
y_train_pred_orig = model_original.predict(X_train_orig)

# Calculate metrics
mse_orig = mean_squared_error(y_test, y_pred_orig)
rmse_orig = np.sqrt(mse_orig)
mae_orig = mean_absolute_error(y_test, y_pred_orig)
r2_orig = r2_score(y_test, y_pred_orig)
r2_train_orig = r2_score(y_train, y_train_pred_orig)

print("\nModel 1 - Original Linear Regression")
print(f"Features: Study_Hours, Previous_Exam_Score, Aptitude_Score")
print(f"Number of features: {X_original.shape[1]}")

print("\nCoefficients:")
for feature, coef in zip(X_original.columns, model_original.coef_):
    print(f"      {feature:<25}: {coef:>10.4f}")
print(f"      {'Intercept':<25}: {model_original.intercept_:>10.4f}")

print("\nPerformance Metrics (Test Set):")
print(f"MSE:  {mse_orig:.4f}")
print(f"RMSE: {rmse_orig:.4f}")
print(f"MAE:  {mae_orig:.4f}")
print(f"R²:   {r2_orig:.4f} (explains {r2_orig*100:.2f}% of variance)")

print("\nPerformance Metrics (Training Set):")
print(f"R²:   {r2_train_orig:.4f}")

"""
STEP 4: Train POLYNOMIAL Multiple Linear Regression (with quadratic term)
"""

print("\n" + "="*80)
print("STEP 4: MODEL 2 - Polynomial Regression (With Study_Hours_Squared)")
print("="*80)

# Prepare features (WITH Study_Hours_Squared)
X_polynomial = data[['Study_Hours', 'Study_Hours_Squared', 
                     'Previous_Exam_Score', 'Aptitude_Score']]

# Split the data (same random state for fair comparison)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_polynomial, y, test_size=0.2, random_state=42
)

# Train the model
model_polynomial = LinearRegression()
model_polynomial.fit(X_train_poly, y_train_poly)

# Make predictions
y_pred_poly = model_polynomial.predict(X_test_poly)
y_train_pred_poly = model_polynomial.predict(X_train_poly)

# Calculate metrics
mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
mae_poly = mean_absolute_error(y_test_poly, y_pred_poly)
r2_poly = r2_score(y_test_poly, y_pred_poly)
r2_train_poly = r2_score(y_train_poly, y_train_pred_poly)

print("\nModel 2 - Polynomial Regression")
print(f"Features: Study_Hours, Study_Hours², Previous_Exam_Score, Aptitude_Score")
print(f"Number of features: {X_polynomial.shape[1]}")

print("\nCoefficients:")
for feature, coef in zip(X_polynomial.columns, model_polynomial.coef_):
    print(f"      {feature:<25}: {coef:>10.4f}")
print(f"      {'Intercept':<25}: {model_polynomial.intercept_:>10.4f}")

print("\nPerformance Metrics (Test Set):")
print(f"MSE:  {mse_poly:.4f}")
print(f"RMSE: {rmse_poly:.4f}")
print(f"MAE:  {mae_poly:.4f}")
print(f"R²:   {r2_poly:.4f} (explains {r2_poly*100:.2f}% of variance)")

print("\nPerformance Metrics (Training Set):")
print(f"R²:   {r2_train_poly:.4f}")

"""
STEP 5: Compare the two models
"""

print("\n" + "="*80)
print("STEP 5: MODEL COMPARISON")
print("="*80)

print("\nSide-by-Side Comparison:")
print(f"\n{'Metric':<20} {'Original Model':<20} {'Polynomial Model':<20} {'Difference':<20}")
print("-" * 80)
print(f"{'R² (Test)':<20} {r2_orig:<20.4f} {r2_poly:<20.4f} {r2_poly - r2_orig:<20.4f}")
print(f"{'R² (Train)':<20} {r2_train_orig:<20.4f} {r2_train_poly:<20.4f} {r2_train_poly - r2_train_orig:<20.4f}")
print(f"{'RMSE':<20} {rmse_orig:<20.4f} {rmse_poly:<20.4f} {rmse_orig - rmse_poly:<20.4f}")
print(f"{'MAE':<20} {mae_orig:<20.4f} {mae_poly:<20.4f} {mae_orig - mae_poly:<20.4f}")
print(f"{'Features Used':<20} {X_original.shape[1]:<20} {X_polynomial.shape[1]:<20} {X_polynomial.shape[1] - X_original.shape[1]:<20}")

# Calculate percentage improvement
r2_improvement = ((r2_poly - r2_orig) / r2_orig) * 100 if r2_orig > 0 else 0
rmse_improvement = ((rmse_orig - rmse_poly) / rmse_orig) * 100 if rmse_orig > 0 else 0

print(f"\nPerformance Improvement:")
print(f"   • R² improved by: {r2_improvement:.2f}%")
print(f"   • RMSE reduced by: {rmse_improvement:.2f}%")
print(f"   • Additional variance explained: {(r2_poly - r2_orig)*100:.2f}%")

"""
STEP 6: Visualize comparison
"""

print("\n" + "="*80)
print("STEP 6: VISUALIZATION COMPARISON")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Comparison: Linear vs Polynomial Regression', 
             fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted - Original Model
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred_orig, alpha=0.6, color='steelblue', 
           edgecolors='black', linewidth=0.5, s=60)
min_val = min(y_test.min(), y_pred_orig.min())
max_val = max(y_test.max(), y_pred_orig.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
ax1.set_xlabel('Actual Score', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted Score', fontsize=11, fontweight='bold')
ax1.set_title('Original Model: Actual vs Predicted', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'R² = {r2_orig:.4f}\nRMSE = {rmse_orig:.2f}', 
        transform=ax1.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Plot 2: Actual vs Predicted - Polynomial Model
ax2 = axes[0, 1]
ax2.scatter(y_test_poly, y_pred_poly, alpha=0.6, color='green', 
           edgecolors='black', linewidth=0.5, s=60)
min_val = min(y_test_poly.min(), y_pred_poly.min())
max_val = max(y_test_poly.max(), y_pred_poly.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
ax2.set_xlabel('Actual Score', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Score', fontsize=11, fontweight='bold')
ax2.set_title('Polynomial Model: Actual vs Predicted', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, f'R² = {r2_poly:.4f}\nRMSE = {rmse_poly:.2f}', 
        transform=ax2.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Plot 3: Residuals - Original Model
ax3 = axes[1, 0]
residuals_orig = y_test - y_pred_orig
ax3.scatter(y_pred_orig, residuals_orig, alpha=0.6, color='coral', 
           edgecolors='black', linewidth=0.5, s=60)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Score', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax3.set_title('Original Model: Residual Plot', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals - Polynomial Model
ax4 = axes[1, 1]
residuals_poly = y_test_poly - y_pred_poly
ax4.scatter(y_pred_poly, residuals_poly, alpha=0.6, color='purple', 
           edgecolors='black', linewidth=0.5, s=60)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Score', fontsize=11, fontweight='bold')
ax4.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax4.set_title('Polynomial Model: Residual Plot', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['R² Score', 'RMSE', 'MAE']
original_vals = [r2_orig, rmse_orig, mae_orig]
polynomial_vals = [r2_poly, rmse_poly, mae_poly]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, original_vals, width, label='Original Model', 
               color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, polynomial_vals, width, label='Polynomial Model', 
               color='green', alpha=0.8, edgecolor='black')

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\nVisualizations created successfully!")

"""
STEP 7: Key insights about linearity assumption
"""

print("\n" + "="*80)
print("STEP 7: KEY INSIGHTS ABOUT THE LINEARITY ASSUMPTION")
print("="*80)

print("\n🔍 What This Exercise Demonstrates:")
print("\n1. LINEAR REGRESSION NAME IS MISLEADING:")
print("   • 'Linear' refers to linearity in COEFFICIENTS, not features")
print("   • We can model non-linear relationships by creating polynomial features")
print("   • The model is still: y = β₀ + β₁x₁ + β₂x₂ + ... (linear in β's)")

print("\n2. IMPORTANCE OF THE LINEARITY ASSUMPTION:")
print("   • If true relationship is quadratic but we use only linear terms:")
print(f"     - We get R² = {r2_orig:.4f} (missing {(r2_poly - r2_orig)*100:.2f}% of explainable variance)")
print("   • When we add the quadratic term:")
print(f"     - We get R² = {r2_poly:.4f} (captures the true relationship better)")

print("\n3. RESIDUAL ANALYSIS IS CRUCIAL:")
print("   • Original model residuals may show patterns (curved or systematic)")
print("   • Polynomial model residuals should be more random (no pattern)")
print("   • Pattern in residuals = violation of linearity assumption")

print("\n4. PRACTICAL IMPLICATIONS:")
print("   • Always plot your data first! Look for non-linear patterns")
print("   • Use residual plots to check if linearity assumption is violated")
print("   • Consider polynomial features when scatter plots show curves")
print("   • Don't overdo it: higher-degree polynomials can overfit")

print("\n5. WHEN TO ADD POLYNOMIAL TERMS:")
print("   • Scatter plots show clear curvature")
print("   • Residual plots show systematic patterns")
print("   • Domain knowledge suggests non-linear relationships")
print("   • Examples: diminishing returns, saturation effects, optimal points")

print("\n6. STILL LINEAR REGRESSION?")
print("   • YES! It's still linear regression because:")
print("   • The model is linear in the PARAMETERS (coefficients)")
print("   • We can still use LinearRegression() from sklearn")
print("   • Formula: Score = β₀ + β₁(Hours) + β₂(Hours²) + β₃(Previous) + β₄(Aptitude)")
print("   •          This is LINEAR in β₀, β₁, β₂, β₃, β₄")

print("\n💡 BOTTOM LINE:")
if r2_improvement > 5:
    print(f"   The {r2_improvement:.1f}% improvement in R² demonstrates that accounting for")
    print("   non-linear relationships is CRUCIAL for accurate predictions!")
else:
    print("   Even small improvements matter in real-world applications.")
    print("   The CONCEPT is important: always check for non-linearity!")

print("\n" + "*"*80)
print("ANALYSIS COMPLETE")
print("*"*80)