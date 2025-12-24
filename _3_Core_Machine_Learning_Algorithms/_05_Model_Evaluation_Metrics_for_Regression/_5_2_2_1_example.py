"""
Exercise 2: Customer Churn Prediction (Regression Component)

While the overall customer churn case study is a classification problem, let's assume for a moment that a sub-problem involves predicting customer lifetime value (CLV) (a continuous variable) instead of churn likelihood. A higher CLV means a customer is expected to generate more revenue over their relationship with the company.

Your model predicts the CLV (in USD) for five customers, and you have their actual observed CLV values:

    Actual CLV (USD): [150, 300, 80, 450, 200]
    Predicted CLV (USD): [160, 280, 90, 420, 210]

Calculate:

    Mean Absolute Error (MAE)
    Mean Squared Error (MSE)
    Root Mean Squared Error (RMSE)
    R-squared (R2R2)

Discuss which metric might be most appropriate for evaluating a CLV prediction model and why. Consider the business implications of over- or under-predicting CLV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style
sns.set_style("whitegrid")

print("="*80)
print("CUSTOMER LIFETIME VALUE (CLV) PREDICTION")
print("="*80)

# data
actual_clv = np.array([150, 300, 80, 450, 200])
predicted_clv = np.array([160, 280, 90, 420, 210])

# Create comprehensive DataFrame
df = pd.DataFrame({
    'Customer': [f'C{i}' for i in range(1, 6)],
    'Actual CLV ($)': actual_clv,
    'Predicted CLV ($)': predicted_clv,
    'Error ($)': actual_clv - predicted_clv,
    'Abs Error ($)': np.abs(actual_clv - predicted_clv),
    'Squared Error ($²)': (actual_clv - predicted_clv) ** 2,
    'Pct Error (%)': ((actual_clv - predicted_clv) / actual_clv * 100).round(2),
    'Prediction Type': ['Over' if p > a else 'Under' for a, p in zip(actual_clv, predicted_clv)]
})

print("\nCustomer Lifetime Value (CLV) Predictions:")
print(df.to_string(index=False))

print(f"\nSummary Statistics:")
print(f"   Actual CLV:")
print(f"      Mean: ${actual_clv.mean():.2f}")
print(f"      Median: ${np.median(actual_clv):.2f}")
print(f"      Std Dev: ${actual_clv.std():.2f}")
print(f"      Range: ${actual_clv.min():.2f} - ${actual_clv.max():.2f}")
print(f"\n   Predicted CLV:")
print(f"      Mean: ${predicted_clv.mean():.2f}")
print(f"      Median: ${np.median(predicted_clv):.2f}")
print(f"      Std Dev: ${predicted_clv.std():.2f}")
print(f"      Range: ${predicted_clv.min():.2f} - ${predicted_clv.max():.2f}")

# Analyze over/under predictions
over_predictions = (predicted_clv > actual_clv).sum()
under_predictions = (predicted_clv < actual_clv).sum()
print(f"\nPrediction Bias:")
print(f"   Over-predictions: {over_predictions} customers")
print(f"   Under-predictions: {under_predictions} customers")
print(f"   Mean bias: ${(predicted_clv - actual_clv).mean():.2f}")

# MAE (Mean Absolute Error)
print("\n" + "="*80)
print("MEAN ABSOLUTE ERROR (MAE)")
print("="*80)

print("\nManual Calculation:")

errors_abs = np.abs(actual_clv - predicted_clv)
print(f"\n   Step 1: Calculate absolute errors for each customer:")
for i, (customer, actual, pred, err) in enumerate(zip(df['Customer'], actual_clv, predicted_clv, errors_abs)):
    print(f"      {customer}: |${actual} - ${pred}| = ${err}")

print(f"\n   Step 2: Sum all absolute errors:")
sum_abs_errors = errors_abs.sum()
print(f"      ${' + '.join(map(str, errors_abs))} = ${sum_abs_errors}")

print(f"\n   Step 3: Divide by number of customers (n = {len(actual_clv)}):")
mae_manual = sum_abs_errors / len(actual_clv)
print(f"      MAE = ${sum_abs_errors} / {len(actual_clv)} = ${mae_manual:.2f}")

# Using sklearn
mae_sklearn = mean_absolute_error(actual_clv, predicted_clv)
print(f"\nVerification with sklearn: MAE = ${mae_sklearn:.2f}")

print(f"\nINTERPRETATION - BUSINESS CONTEXT:")
print(f"   • Average prediction error: ±${mae_manual:.2f} per customer")
print(f"   • Relative error: {(mae_manual/actual_clv.mean())*100:.2f}% of mean CLV")
print(f"   • For 1,000 customers: Total error ≈ ${mae_manual * 1000:,.0f}")
print(f"   • MAE is SYMMETRIC: treats over/under predictions equally")
print(f"   • Easy to interpret: 'On average, we're off by ${mae_manual:.0f}'")


# Calculate MSE (Mean Squared Error)

print("\n" + "="*80)
print("MEAN SQUARED ERROR (MSE)")
print("="*80)

print("\nManual Calculation:")

errors_squared = (actual_clv - predicted_clv) ** 2
print(f"\n   Step 1: Calculate squared errors for each customer:")
for i, (customer, actual, pred, err_sq) in enumerate(zip(df['Customer'], actual_clv, predicted_clv, errors_squared)):
    error = actual - pred
    print(f"      {customer}: (${actual} - ${pred})² = (${error})² = ${err_sq}")

print(f"\n   Step 2: Sum all squared errors:")
sum_squared_errors = errors_squared.sum()
print(f"      ${' + '.join(map(str, errors_squared))} = ${sum_squared_errors}")

print(f"\n   Step 3: Divide by number of customers (n = {len(actual_clv)}):")
mse_manual = sum_squared_errors / len(actual_clv)
print(f"      MSE = ${sum_squared_errors} / {len(actual_clv)} = ${mse_manual:.2f} $²")

# Using sklearn
mse_sklearn = mean_squared_error(actual_clv, predicted_clv)
print(f"\nVerification with sklearn: MSE = ${mse_sklearn:.2f} $²")

print(f"\nINTERPRETATION - BUSINESS CONTEXT:")
print(f"   • MSE = ${mse_manual:.2f} $² (squared units - hard to interpret)")
print(f"   • Heavily penalizes large errors:")
for customer, err, err_sq in zip(df['Customer'], actual_clv - predicted_clv, errors_squared):
    if abs(err) >= 20:
        print(f"      {customer}: Error ${err:+.0f} contributes ${err_sq:.0f} to MSE")
print(f"   • Customer C4 (error=${actual_clv[3]-predicted_clv[3]:+.0f}) dominates the MSE")
print(f"   • Good for optimization, less useful for business interpretation")

# RMSE (Root Mean Squared Error)

print("\n" + "="*80)
print("ROOT MEAN SQUARED ERROR (RMSE)")
print("="*80)

print("\nManual Calculation:")

print(f"\n   Step 1: Take the MSE from Step 3:")
print(f"      MSE = ${mse_manual:.2f} $²")

print(f"\n   Step 2: Take the square root:")
rmse_manual = np.sqrt(mse_manual)
print(f"      RMSE = √${mse_manual:.2f} = ${rmse_manual:.2f}")

# Using sklearn
rmse_sklearn = np.sqrt(mean_squared_error(actual_clv, predicted_clv))
print(f"\nVerification with sklearn: RMSE = ${rmse_sklearn:.2f}")

print(f"\nINTERPRETATION - BUSINESS CONTEXT:")
print(f"   • RMSE = ${rmse_manual:.2f} (same units as CLV - interpretable!)")
print(f"   • Standard deviation of prediction errors")
print(f"   • Compare: MAE = ${mae_manual:.2f}, RMSE = ${rmse_manual:.2f}")
print(f"   • RMSE/MAE ratio = {rmse_manual/mae_manual:.2f}")
print(f"      → Ratio > 1.2 suggests presence of larger errors")
print(f"      → Our ratio indicates some variability in error magnitude")
print(f"   • Relative RMSE: {(rmse_manual/actual_clv.mean())*100:.2f}% of mean CLV")
print(f"   • Interpretation: 'Typical uncertainty is ±${rmse_manual:.0f} per customer'")

# R² (R-squared)

print("\n" + "="*80)
print("R-SQUARED (R² / Coefficient of Determination)")
print("="*80)

print("\nManual Calculation:")

# Calculate SS_res
ss_res = errors_squared.sum()
print(f"\n   Step 1: Calculate SS_res (Residual Sum of Squares):")
print(f"      SS_res = Σ(actual - predicted)²")
print(f"      SS_res = ${ss_res:.2f}")

# Calculate SS_tot
y_mean = actual_clv.mean()
ss_tot_components = (actual_clv - y_mean) ** 2
ss_tot = ss_tot_components.sum()

print(f"\n   Step 2: Calculate SS_tot (Total Sum of Squares):")
print(f"      Mean actual CLV = ${y_mean:.2f}")
print(f"      SS_tot = Σ(actual - mean)²")
for customer, actual, component in zip(df['Customer'], actual_clv, ss_tot_components):
    print(f"         {customer}: (${actual} - ${y_mean:.2f})² = ${component:.2f}")
print(f"      SS_tot = ${ss_tot:.2f}")

print(f"\n   Step 3: Calculate R²:")
r2_manual = 1 - (ss_res / ss_tot)
print(f"      R² = 1 - (SS_res / SS_tot)")
print(f"      R² = 1 - (${ss_res:.2f} / ${ss_tot:.2f})")
print(f"      R² = 1 - {ss_res/ss_tot:.4f}")
print(f"      R² = {r2_manual:.4f}")

# Using sklearn
r2_sklearn = r2_score(actual_clv, predicted_clv)
print(f"\nVerification with sklearn: R² = {r2_sklearn:.4f}")

print(f"\nINTERPRETATION - BUSINESS CONTEXT:")
print(f"   • R² = {r2_manual:.4f} or {r2_manual*100:.2f}%")
print(f"   • Model explains {r2_manual*100:.2f}% of variance in CLV")
print(f"   • {(1-r2_manual)*100:.2f}% of variance remains unexplained")
print(f"\n   CLV Prediction Quality Scale:")
print(f"   ✅ R² > 0.90 → Excellent (our model: {r2_manual:.4f})")
print(f"   ✓  R² 0.70-0.90 → Good")
print(f"   ⚠  R² 0.50-0.70 → Moderate")
print(f"   ❌ R² < 0.50 → Poor")
print(f"\n   Business Impact:")
print(f"   • High R² means reliable customer segmentation")
print(f"   • Can confidently allocate marketing resources")
print(f"   • Small unexplained variance likely due to random factors")

# Summary and Comparison

print("\n" + "="*80)
print("METRICS SUMMARY & COMPARISON")
print("="*80)

summary = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R²', 'RMSE/MAE Ratio'],
    'Value': [mae_manual, mse_manual, rmse_manual, r2_manual, rmse_manual/mae_manual],
    'Unit': ['$', '$²', '$', 'unitless', 'ratio'],
    'Interpretation': [
        f'Avg error: ±${mae_manual:.2f}',
        f'Avg squared error: ${mse_manual:.2f}',
        f'Std dev of errors: ±${rmse_manual:.2f}',
        f'Explains {r2_manual*100:.2f}% variance',
        f'Error variability indicator'
    ]
})

print("\nSummary Table:")
print(summary.to_string(index=False))

print(f"\nRelative Metrics:")
print(f"   • MAE/Mean CLV: {(mae_manual/actual_clv.mean())*100:.2f}%")
print(f"   • RMSE/Mean CLV: {(rmse_manual/actual_clv.mean())*100:.2f}%")
print(f"   • Typical error relative to average customer value: ~{(mae_manual/actual_clv.mean())*100:.0f}%")

# Visualizations

print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Customer Lifetime Value (CLV) Prediction - Comprehensive Analysis', 
             fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted (Bar Chart)
ax1 = fig.add_subplot(gs[0, :2])
customers = df['Customer']
x = np.arange(len(customers))
width = 0.35

bars1 = ax1.bar(x - width/2, actual_clv, width, label='Actual CLV', 
                color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, predicted_clv, width, label='Predicted CLV', 
                color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.0f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

ax1.set_xlabel('Customer', fontsize=12, fontweight='bold')
ax1.set_ylabel('CLV (USD)', fontsize=12, fontweight='bold')
ax1.set_title('Actual vs Predicted CLV by Customer', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(customers)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Metrics Summary
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
metrics_text = f"""
PERFORMANCE METRICS

MAE: ${mae_manual:.2f}
RMSE: ${rmse_manual:.2f}
R²: {r2_manual:.4f}

Relative Error:
  {(mae_manual/actual_clv.mean())*100:.2f}% of mean CLV

Model Quality:
  EXCELLENT ✓
  
Variance Explained:
  {r2_manual*100:.2f}%
"""
ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes,
        fontsize=11, verticalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Plot 3: Scatter Plot
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(actual_clv, predicted_clv, s=200, alpha=0.7, color='green', 
           edgecolors='black', linewidth=2)
min_val = min(actual_clv.min(), predicted_clv.min()) - 20
max_val = max(actual_clv.max(), predicted_clv.max()) + 20
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, 
        label='Perfect prediction', alpha=0.8)

for i, customer in enumerate(customers):
    ax3.annotate(customer, (actual_clv[i], predicted_clv[i]), 
                xytext=(7, 7), textcoords='offset points',
                fontsize=10, fontweight='bold')

ax3.set_xlabel('Actual CLV ($)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted CLV ($)', fontsize=11, fontweight='bold')
ax3.set_title(f'Actual vs Predicted (R² = {r2_manual:.4f})', 
             fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Error Distribution
ax4 = fig.add_subplot(gs[1, 1])
errors = actual_clv - predicted_clv
colors_err = ['red' if e < 0 else 'green' for e in errors]
bars = ax4.bar(customers, errors, color=colors_err, alpha=0.7, 
              edgecolor='black', linewidth=1.5)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=2)

for bar, err in zip(bars, errors):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'${err:+.0f}', ha='center', 
            va='bottom' if height > 0 else 'top', 
            fontsize=10, fontweight='bold')

ax4.set_xlabel('Customer', fontsize=11, fontweight='bold')
ax4.set_ylabel('Error (Actual - Predicted) $', fontsize=11, fontweight='bold')
ax4.set_title('Prediction Errors (Green=Under, Red=Over)', 
             fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Absolute Errors
ax5 = fig.add_subplot(gs[1, 2])
bars = ax5.barh(customers, abs(errors), color='orange', alpha=0.7, 
               edgecolor='black', linewidth=1.5)

for bar, err in zip(bars, abs(errors)):
    width = bar.get_width()
    ax5.text(width, bar.get_y() + bar.get_height()/2.,
            f' ${err:.0f}', ha='left', va='center', 
            fontsize=10, fontweight='bold')

ax5.axvline(x=mae_manual, color='red', linestyle='--', linewidth=2, 
           label=f'MAE = ${mae_manual:.2f}')
ax5.set_xlabel('Absolute Error ($)', fontsize=11, fontweight='bold')
ax5.set_title('Absolute Errors by Customer', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Percentage Errors
ax6 = fig.add_subplot(gs[2, 0])
pct_errors = ((actual_clv - predicted_clv) / actual_clv * 100)
colors_pct = ['red' if e < 0 else 'green' for e in pct_errors]
bars = ax6.bar(customers, pct_errors, color=colors_pct, alpha=0.7, 
              edgecolor='black', linewidth=1.5)
ax6.axhline(y=0, color='black', linestyle='-', linewidth=2)

for bar, pct in zip(bars, pct_errors):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:+.1f}%', ha='center', 
            va='bottom' if height > 0 else 'top', 
            fontsize=9, fontweight='bold')

ax6.set_xlabel('Customer', fontsize=11, fontweight='bold')
ax6.set_ylabel('Percentage Error (%)', fontsize=11, fontweight='bold')
ax6.set_title('Relative Prediction Errors', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Error Metrics Comparison
ax7 = fig.add_subplot(gs[2, 1])
metric_names = ['MAE', 'RMSE']
metric_values = [mae_manual, rmse_manual]
colors_met = ['steelblue', 'coral']

bars = ax7.barh(metric_names, metric_values, color=colors_met, 
               alpha=0.7, edgecolor='black', linewidth=1.5)

for i, (name, value) in enumerate(zip(metric_names, metric_values)):
    ax7.text(value, i, f'  ${value:.2f}', va='center', 
            fontsize=11, fontweight='bold')

ax7.set_xlabel('Error ($)', fontsize=11, fontweight='bold')
ax7.set_title('Error Metrics Comparison', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='x')

# Plot 8: Business Impact
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

# Calculate business metrics
total_actual = actual_clv.sum()
total_predicted = predicted_clv.sum()
total_abs_error = abs(actual_clv - predicted_clv).sum()

impact_text = f"""
BUSINESS IMPACT

Total Portfolio:
  Actual: ${total_actual:,.0f}
  Predicted: ${total_predicted:,.0f}
  Difference: ${total_predicted - total_actual:+,.0f}

At 10,000 customers:
  Error: ±${mae_manual * 10000:,.0f}
  
Decision Quality:
  R² = {r2_manual:.4f}
  → HIGH CONFIDENCE ✓
  
Resource Allocation:
  Reliable for strategic
  planning decisions
"""

ax8.text(0.1, 0.5, impact_text, transform=ax8.transAxes,
        fontsize=10, verticalalignment='center', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.savefig('clv_prediction_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Which Metric is Most Appropriate for CLV?

print("\n" + "="*80)
print("WHICH METRIC IS MOST APPROPRIATE FOR CLV PREDICTION?")
print("="*80)

print("""
1️⃣ THE WINNER: MAE (Mean Absolute Error) - MOST APPROPRIATE ✓

ADVANTAGES:
   • Direct dollar interpretation: "We're off by ${mae:.0f} on average"
   • Easy to communicate to business stakeholders
   • Symmetric: treats over/under predictions equally
   • Robust to outliers (linear penalty)
   • Aligns with business KPIs (revenue impact)
   
Business Translation:
   • "Marketing budget might be off by ${mae:.0f} per customer"
   • "For 10,000 customers, expect ±${mae_10k_error:,.0f} total error"
   • Clear ROI calculations possible
   
Use Cases:
   • Budget planning and forecasting
   • Customer segmentation decisions
   • Resource allocation strategies
   • Marketing campaign targeting

---------------------------------------------------------------------------------------

2️⃣  RMSE (${rmse:.2f}) - GOOD SECONDARY METRIC
---------------------------------------------------------------------------------------
   
ADVANTAGES:
   • Same units as CLV (interpretable)
   • Penalizes large errors (important for high-value customers)
   • Shows prediction variability
   
DISADVANTAGES:
   • Higher sensitivity to outliers
   • Less intuitive than MAE for stakeholders
   
When to Prioritize:
   • When large errors are MUCH worse than small errors
   • Example: Losing a $450 customer (C4) is catastrophic
   • High-stakes decisions where worst-case matters
   
Use Cases:
   • Risk assessment
   • VIP customer identification
   • Churn prevention for high-value segments

---------------------------------------------------------------------------------------

3️⃣  R² ({r2:.4f}) - EXCELLENT COMPLEMENTARY METRIC
---------------------------------------------------------------------------------------
   
ADVANTAGES:
   • Shows model quality ({r2_pct:.1f}% variance explained)
   • Standardized (0 to 1 scale)
   • Good for model comparison
   
LIMITATIONS:
   • Doesn't show magnitude of errors in dollars
   • High R² doesn't guarantee acceptable business error
   
Best For:
   • Comparing different models
   • Academic/technical reporting
   • Understanding model fit quality
   
Use Cases:
   • Model selection and validation
   • A/B testing different algorithms
   • Technical documentation

---------------------------------------------------------------------------------------

4️⃣  MSE (${mse:.2f} $²) - LEAST USEFUL FOR BUSINESS
---------------------------------------------------------------------------------------
   
DISADVANTAGES:
   • Squared units ($²) - not interpretable
   • Cannot directly relate to business impact
   • Only useful for optimization
   
When to Use:
   • Training machine learning models (loss function)
   • Mathematical optimization
   • Never for business reporting!

---------------------------------------------------------------------------------------

OVER-PREDICTING CLV (Predicted > Actual)
---------------------------------------------------------------------------------------
   
Our Data: {over_predictions} customers over-predicted
   
RISKS:
   • Excessive marketing spend on low-value customers
   • Over-allocation of customer service resources
   • Inflated revenue forecasts
   • Disappointed stakeholders when actual revenue lower
   • Poor ROI on retention campaigns
   
Cost Example:
   • Spend $50 on retention for predicted $280 customer (C2)
   • Actual value only $280 → Still profitable
   • BUT: Could have spent less or focused elsewhere
   
Strategic Impact:
   • Moderate risk: Wastes resources but doesn't lose customers
   • Can lead to bloated customer acquisition costs (CAC)
   • May mask profitability issues

---------------------------------------------------------------------------------------

UNDER-PREDICTING CLV (Predicted < Actual)
---------------------------------------------------------------------------------------
   
Our Data: {under_predictions} customers under-predicted
   
RISKS (OFTEN WORSE!):
   • Under-investment in high-value customers
   • Missed retention opportunities
   • Competitors may steal valuable customers
   • Lost lifetime revenue
   • Damaged customer relationships
Cost Example:
   • Predict C4 worth $420, actually worth $450
   • Under-invest by $30 in retention
   • Risk losing $450 customer entirely!
   • Loss >> savings from under-spending

Strategic Impact:
   • HIGH RISK: Can lose valuable customers permanently
   • Opportunity cost of not maximizing high-value relationships
   • Competitive disadvantage

---------------------------------------------------------------------------------------

ASYMMETRIC COSTS: Which Error is Worse?
---------------------------------------------------------------------------------------

Generally: UNDER-PREDICTION IS WORSE

Reasoning:
    • Over-prediction: Waste money (recoverable)
    • Under-prediction: Lose customers (permanent)

Cost Comparison:
    • Over-predict $280 customer by $20:
        → Spend extra $20 on retention
        → Loss: $20 (one-time)
    • Under-predict $450 customer by $30:
        → Under-invest, customer churns
        → Loss: $450 (entire lifetime value!)
    Asymmetry Ratio: 450/20 = 22.5x worse!

---------------------------------------------------------------------------------------

RECOMMENDED METRIC STRATEGY FOR CLV
---------------------------------------------------------------------------------------
PRIMARY METRIC: MAE
    • Track: "Average error is ±${mae:.0f}"
    • Goal: Keep MAE < 10% of mean CLV
    • Current: {mae_rel_pct:.2f}% ✓ EXCELLENT
SECONDARY METRIC: RMSE
    • Monitor for large errors
    • Alert if RMSE/MAE > 1.5 (outlier indicator)
    • Current: {rmse_mae_ratio:.2f} ✓ ACCEPTABLE
TERTIARY METRIC: R²
    • Overall model quality check
    • Ensure R² > 0.80 for production use
    • Current: {r2:.4f} ✓ EXCELLENT
DIRECTIONAL BIAS CHECK:
    • Monitor: mean(predicted - actual)
    • Current: ${mean_bias:.2f}
    • If consistently negative → Under-predicting (HIGH RISK!)
    • If consistently positive → Over-predicting (Moderate risk)

---------------------------------------------------------------------------------------

💼 PRACTICAL RECOMMENDATIONS
---------------------------------------------------------------------------------------

Use MAE for executive reporting
    → "Our CLV predictions are accurate within ±${mae:.0f}"
Use RMSE for technical teams
    → "Monitor for prediction volatility"
Use R² for model selection
    → "Compare different algorithms"
Consider asymmetric loss functions
    → Penalize under-predictions more heavily
    → Example: Loss = |error| if over, 2×|error| if under
Segment analysis by CLV tier
    → Low CLV ($0-$200): MAE acceptable
    → High CLV ($300+): Use RMSE, zero tolerance
Set business-driven thresholds
    → Acceptable MAE: 5-10% of mean CLV
    → Current: {mae_rel_pct:.2f}% ✓
    → Action: No changes needed!

---------------------------------------------------------------------------------------
FINAL VERDICT
---------------------------------------------------------------------------------------
BEST METRIC: MAE (Mean Absolute Error)
WHY?
    ✓ Dollar-interpretable (${mae:.2f})
    ✓ Easy stakeholder communication
    ✓ Aligns with business goals
    ✓ Robust and reliable
    ✓ Directly measures impact

SUPPLEMENTARY METRICS:
    RMSE: For large error detection
    R²: For model quality assessment
    Directional bias: For systematic error detection

MODEL VERDICT:
    With MAE = ${mae:.2f} ({mae_rel_pct:.2f}% of mean CLV)
    and R² = {r2:.4f}
    → This model is EXCELLENT for production use!
""".format(
    mae=mae_manual,
    rmse=rmse_manual,
    mse=mse_manual,
    r2=r2_manual,
    over_predictions=over_predictions,
    under_predictions=under_predictions,
    mae_10k_error=mae_manual * 10000,
    r2_pct=r2_manual * 100,
    mae_rel_pct=(mae_manual/actual_clv.mean())*100,
    rmse_mae_ratio=rmse_manual/mae_manual,
    mean_bias=(predicted_clv - actual_clv).mean()
))