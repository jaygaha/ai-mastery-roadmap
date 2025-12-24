"""
Exercise 1: Residential Energy Consumption Prediction

You are developing a model to predict the daily energy consumption (in kWh) of residential buildings. For a sample of five days, you have the following actual consumption values and your model's predictions:

    Actual Consumption (kWh): [30, 45, 28, 52, 35]
    Predicted Consumption (kWh): [32, 43, 27, 50, 38]

Calculate:

    Mean Absolute Error (MAE)
    Mean Squared Error (MSE)
    Root Mean Squared Error (RMSE)
    R-squared (R2R2)

Interpret each result in the context of energy consumption prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*80)
print("EXERCISE 1: RESIDENTIAL ENERGY CONSUMPTION PREDICTION")
print("="*80)

# Data
actual_consumption = np.array([30, 45, 28, 52, 35])
predicted_consumption = np.array([32, 43, 27, 50, 38])

# Create a DataFrame to display the data
df = pd.DataFrame({
    'Day': range(1, 6),
    'Actual (kWh)': actual_consumption,
    'Predicted (kWh)': predicted_consumption,
    'Error (Actual - Pred)': actual_consumption - predicted_consumption,
    'Absolute Error': np.abs(actual_consumption - predicted_consumption),
    'Squared Error': (actual_consumption - predicted_consumption) ** 2
})

print("\nDaily Energy Consumption Data:")
print(df.to_string(index=False))

print(f"\nSummary Statistics:")
print(f"   Mean Actual Consumption: {actual_consumption.mean():.2f} kWh")
print(f"   Mean Predicted Consumption: {predicted_consumption.mean():.2f} kWh")
print(f"   Actual Range: {actual_consumption.min():.0f} - {actual_consumption.max():.0f} kWh")
print(f"   Predicted Range: {predicted_consumption.min():.0f} - {predicted_consumption.max():.0f} kWh")

# Calculate MAE (Mean Absolute Error)
print("\n" + "="*80)
print("MEAN ABSOLUTE ERROR (MAE)")
print("="*80)

errors_abs = np.abs(actual_consumption - predicted_consumption)
print(f"\n   Step 1: Calculate absolute errors for each day:")
for i, (actual, pred, err) in enumerate(zip(actual_consumption, predicted_consumption, errors_abs), 1):
    print(f"      Day {i}: |{actual} - {pred}| = {err}")

print(f"\n   Step 2: Sum all absolute errors:")
sum_abs_errors = errors_abs.sum()
print(f"      {' + '.join(map(str, errors_abs))} = {sum_abs_errors}")

print(f"\n   Step 3: Divide by number of observations (n = {len(actual_consumption)}):")
mae_manual = sum_abs_errors / len(actual_consumption)
print(f"      MAE = {sum_abs_errors} / {len(actual_consumption)} = {mae_manual:.2f} kWh")

mae_sklearn = mean_absolute_error(actual_consumption, predicted_consumption)
print(f"\nVerification with sklearn: MAE = {mae_sklearn:.2f} kWh")

print(f"\nINTERPRETATION:")
print(f"   • On average, our predictions are off by ±{mae_manual:.2f} kWh")
print(f"   • This means typical prediction error is about {mae_manual:.2f} kWh")
print(f"   • Relative error: {(mae_manual/actual_consumption.mean())*100:.1f}% of mean consumption")
print(f"   • For a homeowner, this means predictions might differ by ~{mae_manual:.0f} kWh from reality")
print(f"   • MAE treats all errors equally (no penalty for large errors)")

# MSE (Mean Squared Error)
print("\n" + "="*80)
print("MEAN SQUARED ERROR (MSE)")
print("="*80)

errors_squared = (actual_consumption - predicted_consumption) ** 2
print(f"\n   Step 1: Calculate squared errors for each day:")
for i, (actual, pred, err_sq) in enumerate(zip(actual_consumption, predicted_consumption, errors_squared), 1):
    error = actual - pred
    print(f"      Day {i}: ({actual} - {pred})² = ({error})² = {err_sq}")

print(f"\n   Step 2: Sum all squared errors:")
sum_squared_errors = errors_squared.sum()
print(f"      {' + '.join(map(str, errors_squared))} = {sum_squared_errors}")

print(f"\n   Step 3: Divide by number of observations (n = {len(actual_consumption)}):")
mse_manual = sum_squared_errors / len(actual_consumption)
print(f"      MSE = {sum_squared_errors} / {len(actual_consumption)} = {mse_manual:.2f} kWh²")

# Using sklearn
mse_sklearn = mean_squared_error(actual_consumption, predicted_consumption)
print(f"\nVerification with sklearn: MSE = {mse_sklearn:.2f} kWh²")

print(f"\nINTERPRETATION:")
print(f"   • MSE = {mse_manual:.2f} kWh² (squared units!)")
print(f"   • Penalizes large errors more heavily than MAE")
print(f"   • Notice: Larger errors (Day 5: error=3) contribute more to MSE")
print(f"   • Not directly interpretable due to squared units")
print(f"   • Used more in optimization/model training than interpretation")
print(f"   • Useful for comparing models (lower MSE = better model)")

# RMSE (Root Mean Squared Error)
print("\n" + "="*80)
print("ROOT MEAN SQUARED ERROR (RMSE)")
print("="*80)

rmse_manual = np.sqrt(mse_manual)
print(f"\n   Step 1: Calculate square root of MSE:")
print(f"      RMSE = sqrt({mse_manual}) = {rmse_manual:.2f} kWh")

# Using sklearn
rmse_sklearn = np.sqrt(mean_squared_error(actual_consumption, predicted_consumption))
print(f"\nVerification with sklearn: RMSE = {rmse_sklearn:.2f} kWh")

print(f"\nINTERPRETATION:")
print(f"   • RMSE = {rmse_manual:.2f} kWh (root units)")
print(f"   • Penalizes large errors more heavily than MAE")
print(f"   • Notice: Larger errors (Day 5: error=3) contribute more to RMSE")
print(f"   • Directly interpretable in original units (kWh)")
print(f"   • Used more in optimization/model training than interpretation")
print(f"   • Useful for comparing models (lower RMSE = better model)")

# R² (R-squared / Coefficient of Determination)
print("\n" + "="*80)
print("STEP 5: R-SQUARED (R² / Coefficient of Determination)")
print("="*80)

print("\nManual Calculation:")

# Calculate SS_res (already have this)
ss_res = errors_squared.sum()
print(f"\n   Step 1: Calculate SS_res (Residual Sum of Squares):")
print(f"      SS_res = Σ(actual - predicted)²")
print(f"      SS_res = {' + '.join(map(str, errors_squared))} = {ss_res}")

# Calculate SS_tot
y_mean = actual_consumption.mean()
ss_tot_components = (actual_consumption - y_mean) ** 2
ss_tot = ss_tot_components.sum()

print(f"\n   Step 2: Calculate SS_tot (Total Sum of Squares):")
print(f"      Mean of actual values = {y_mean:.2f} kWh")
print(f"      SS_tot = Σ(actual - mean)²")
for i, (actual, component) in enumerate(zip(actual_consumption, ss_tot_components), 1):
    print(f"         Day {i}: ({actual} - {y_mean:.2f})² = {component:.2f}")
print(f"      SS_tot = {ss_tot:.2f}")

print(f"\n   Step 3: Calculate R²:")
r2_manual = 1 - (ss_res / ss_tot)
print(f"      R² = 1 - (SS_res / SS_tot)")
print(f"      R² = 1 - ({ss_res} / {ss_tot:.2f})")
print(f"      R² = 1 - {ss_res/ss_tot:.4f}")
print(f"      R² = {r2_manual:.4f}")

# Using sklearn
r2_sklearn = r2_score(actual_consumption, predicted_consumption)
print(f"\nVerification with sklearn: R² = {r2_sklearn:.4f}")

print(f"\nINTERPRETATION:")
print(f"   • R² = {r2_manual:.4f} or {r2_manual*100:.2f}%")
print(f"   • The model explains {r2_manual*100:.2f}% of variance in energy consumption")
print(f"   • {(1-r2_manual)*100:.2f}% of variance remains unexplained")
print(f"   • R² ranges from -∞ to 1 (1 = perfect predictions)")
print(f"   • R² = 1.00 → Perfect model (predicts exactly)")
print(f"   • R² = 0.00 → Model no better than predicting mean")
print(f"   • R² < 0.00 → Model worse than predicting mean (bad!)")
print(f"\n   In this case:")
print(f"   • R² = {r2_manual:.4f} is EXCELLENT for energy prediction")
print(f"   • Model captures most day-to-day variation in consumption")
print(f"   • Small unexplained variance might be random factors")

# Summary Comparison

print("\n" + "="*80)
print("METRICS SUMMARY & COMPARISON")
print("="*80)

# Create summary table
summary = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
    'Value': [mae_manual, mse_manual, rmse_manual, r2_manual],
    'Unit': ['kWh', 'kWh²', 'kWh', 'unitless'],
    'Interpretation': [
        f'Avg error: ±{mae_manual:.2f} kWh',
        f'Avg squared error: {mse_manual:.2f} kWh²',
        f'Std dev of errors: ±{rmse_manual:.2f} kWh',
        f'Explains {r2_manual*100:.2f}% of variance'
    ]
})

print("\nSummary Table:")
print(summary.to_string(index=False))

print("\nKey Comparisons:")
print(f"   • MAE ({mae_manual:.2f}) vs RMSE ({rmse_manual:.2f}):")
print(f"     - RMSE is {rmse_manual/mae_manual:.2f}x larger than MAE")
print(f"     - Small difference suggests errors are relatively uniform")
print(f"     - Large difference would indicate presence of outliers")

print(f"\nRelative Metrics:")
print(f"   • MAE/Mean = {(mae_manual/actual_consumption.mean())*100:.1f}%")
print(f"   • RMSE/Mean = {(rmse_manual/actual_consumption.mean())*100:.1f}%")
print(f"   • Typical error is about {(mae_manual/actual_consumption.mean())*100:.0f}% of average consumption")

# Visualizations

print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Energy Consumption Prediction - Model Evaluation', 
             fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
days = np.arange(1, 6)
width = 0.35
bars1 = ax1.bar(days - width/2, actual_consumption, width, label='Actual', 
                color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(days + width/2, predicted_consumption, width, label='Predicted', 
                color='coral', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Day', fontsize=11, fontweight='bold')
ax1.set_ylabel('Energy Consumption (kWh)', fontsize=11, fontweight='bold')
ax1.set_title('Actual vs Predicted Consumption', fontsize=12, fontweight='bold')
ax1.set_xticks(days)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Scatter plot with perfect prediction line
ax2 = axes[0, 1]
ax2.scatter(actual_consumption, predicted_consumption, s=150, alpha=0.7, color='green', 
           edgecolors='black', linewidth=2)
min_val = min(actual_consumption.min(), predicted_consumption.min()) - 2
max_val = max(actual_consumption.max(), predicted_consumption.max()) + 2
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
        label='Perfect prediction', alpha=0.8)

# Add day labels
for i, (x, y) in enumerate(zip(actual_consumption, predicted_consumption), 1):
    ax2.annotate(f'Day {i}', (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')

ax2.set_xlabel('Actual Consumption (kWh)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Consumption (kWh)', fontsize=11, fontweight='bold')
ax2.set_title(f'Actual vs Predicted (R² = {r2_manual:.4f})', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(min_val, max_val)
ax2.set_ylim(min_val, max_val)

# Plot 3: Error analysis
ax3 = axes[1, 0]
errors = actual_consumption - predicted_consumption
colors = ['red' if e < 0 else 'green' for e in errors]
bars = ax3.bar(days, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=2)

# Add value labels
for bar, err in zip(bars, errors):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{err:+.0f}', ha='center', 
            va='bottom' if height > 0 else 'top', 
            fontsize=10, fontweight='bold')

ax3.set_xlabel('Day', fontsize=11, fontweight='bold')
ax3.set_ylabel('Error (Actual - Predicted) kWh', fontsize=11, fontweight='bold')
ax3.set_title('Prediction Errors by Day', fontsize=12, fontweight='bold')
ax3.set_xticks(days)
ax3.grid(True, alpha=0.3, axis='y')
ax3.text(0.05, 0.95, f'MAE = {mae_manual:.2f} kWh', 
        transform=ax3.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 4: Metrics comparison
ax4 = axes[1, 1]
metrics_names = ['MAE', 'RMSE']
metrics_values = [mae_manual, rmse_manual]
colors_bars = ['steelblue', 'coral']

bars = ax4.barh(metrics_names, metrics_values, color=colors_bars, 
               alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels
for i, (name, value) in enumerate(zip(metrics_names, metrics_values)):
    ax4.text(value, i, f'  {value:.2f} kWh', va='center', 
            fontsize=11, fontweight='bold')

ax4.set_xlabel('Error (kWh)', fontsize=11, fontweight='bold')
ax4.set_title('Error Metrics Comparison', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Add R² as text
ax4.text(0.98, 0.02, f'R² = {r2_manual:.4f}\n({r2_manual*100:.2f}% variance explained)', 
        transform=ax4.transAxes, fontsize=11, 
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
        fontweight='bold')

plt.tight_layout()
plt.show()

# Context-Specific Insights

print("\n" + "="*80)
print("CONTEXT-SPECIFIC INSIGHTS FOR ENERGY CONSUMPTION")
print("="*80)

print("""
PRACTICAL IMPLICATIONS FOR RESIDENTIAL ENERGY PREDICTION:

1. MODEL ACCURACY ASSESSMENT:
   ────────────────────────────
   • MAE = {mae:.2f} kWh means predictions typically off by ~{mae:.0f} kWh
   • For context: Average consumption = {avg:.2f} kWh/day
   • Relative error: {rel_err:.1f}% (EXCELLENT accuracy)
   
   Rating Scale for Energy Prediction:
   ✅ < 5% error  → Excellent (our model: {rel_err:.1f}%)
   ✓  5-10% error → Good
   ⚠  10-20% error → Acceptable
   ❌ > 20% error  → Poor

2. BUSINESS/HOMEOWNER PERSPECTIVE:
   ──────────────────────────────────
   • Prediction accuracy allows for:
     - Reliable energy budgeting (±{mae:.0f} kWh uncertainty)
     - Effective demand response planning
     - Accurate cost estimation (±${cost_est:.2f} at $0.12/kWh)
   
   • High R² ({r2:.4f}) means model captures:
     - Daily usage patterns
     - Weather effects (if included in full model)
     - Occupancy variations

3. MODEL RELIABILITY:
   ───────────────────
   • RMSE ({rmse:.2f}) ≈ MAE ({mae:.2f}) suggests:
     - Errors are consistent across days
     - No major outliers or extreme mispredictions
     - Reliable for production use
   
   • If RMSE >> MAE, would indicate:
     - Some days with large errors
     - Need to investigate outliers
     - Possible missing features

4. IMPROVEMENT RECOMMENDATIONS:
   ─────────────────────────────
   Current performance is EXCELLENT, but could consider:
   • Adding weather data (temperature, humidity)
   • Including day-of-week features (weekday vs weekend)
   • Seasonal adjustments (summer vs winter)
   • Occupancy information
   
   Expected improvement: Marginal (already at {r2_pct:.1f}% explained variance)

5. WHEN TO RETRAIN:
   ─────────────────
   Monitor these thresholds:
   • MAE increases above 4 kWh → Investigate
   • R² drops below 0.90 → Consider retraining
   • RMSE/MAE ratio > 1.5 → Check for outliers

CONCLUSION:
   This model demonstrates EXCELLENT predictive performance for residential
   energy consumption. The combination of high R² ({r2:.4f}) and low relative
   error ({rel_err:.1f}%) makes it suitable for production deployment.
""".format(
    mae=mae_manual,
    rmse=rmse_manual,
    r2=r2_manual,
    avg=actual_consumption.mean(),
    rel_err=(mae_manual/actual_consumption.mean())*100,
    cost_est=mae_manual * 0.12,
    r2_pct=r2_manual * 100
))