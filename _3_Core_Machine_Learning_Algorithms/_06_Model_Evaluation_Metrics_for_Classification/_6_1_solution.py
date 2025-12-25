"""
Scenario 1: Fraud Detection Model

    True Positives (TP): 95 (Correctly identified fraudulent transactions)
    True Negatives (TN): 9800 (Correctly identified legitimate transactions)
    False Positives (FP): 5 (Legitimate transactions incorrectly flagged as fraud)
    False Negatives (FN): 10 (Fraudulent transactions missed)

    Calculate Accuracy.
    Calculate Precision.
    Calculate Recall.
    Calculate F1-Score.
    Based on these metrics, discuss whether this model prioritizes avoiding false alarms or catching all fraudulent activities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

print("="*80)
print("FRAUD DETECTION MODEL - CLASSIFICATION METRICS")
print("="*80)

# The given values
TP = 95   # True Positives: Fraud that we correctly caught
TN = 9800 # True Negatives: Legitimate transactions we correctly identified
FP = 5    # False Positives: Legitimate transactions we wrongly flagged as fraud
FN = 10   # False Negatives: Fraud transactions we missed

total = TP + TN + FP + FN  # Total transactions = 9910
predicted_fraud = TP + FP

# Calculate metrics
accuracy = (TP + TN) / total
precision = TP / predicted_fraud
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)

print("\n" + "="*80)
print("ALL METRICS SUMMARY")
print("="*80)

summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [f'{accuracy*100:.2f}%', f'{precision*100:.2f}%', 
              f'{recall*100:.2f}%', f'{f1_score*100:.2f}%'],
    'Simple Explanation': [
        'How often we are correct overall',
        'When we say fraud, how often are we right?',
        'Of all fraud, how much do we catch?',
        'Balance between precision and recall'
    ]
})

print("\nComplete Summary:")
print(summary.to_string(index=False))

print("\n" + "="*80)
print("VISUAL SUMMARY")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fraud Detection Model - Easy-to-Understand Metrics', 
             fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
ax1 = axes[0, 0]
conf_data = [[TN, FP], [FN, TP]]
sns.heatmap(conf_data, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Predicted\nLegitimate', 'Predicted\nFraud'],
            yticklabels=['Actual\nLegitimate', 'Actual\nFraud'],
            cbar=False, annot_kws={'size': 16, 'weight': 'bold'})
ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold')

# Add labels
ax1.text(0.5, 0.3, 'Correct\n✓', ha='center', va='center', 
        fontsize=11, color='darkblue', weight='bold')
ax1.text(1.5, 0.3, 'False Alarm\n❌', ha='center', va='center', 
        fontsize=11, color='darkred', weight='bold')
ax1.text(0.5, 1.3, 'Missed\n❌', ha='center', va='center', 
        fontsize=11, color='darkred', weight='bold')
ax1.text(1.5, 1.3, 'Correct\n✓', ha='center', va='center', 
        fontsize=11, color='darkgreen', weight='bold')

# Plot 2: Metrics Bar Chart
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy*100, precision*100, recall*100, f1_score*100]
colors = ['steelblue', 'green', 'orange', 'purple']

bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('All Metrics at a Glance', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: What We Got Right vs Wrong
ax3 = axes[1, 0]
outcomes = ['Correct\nPredictions', 'Wrong\nPredictions']
outcome_vals = [TP + TN, FP + FN]
outcome_colors = ['green', 'red']

bars = ax3.bar(outcomes, outcome_vals, color=outcome_colors, alpha=0.7, 
              edgecolor='black', linewidth=2)
for bar, val in zip(bars, outcome_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:,}\n({val/total*100:.2f}%)', ha='center', va='bottom', 
            fontsize=11, fontweight='bold')

ax3.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
ax3.set_title(f'Overall: {accuracy*100:.2f}% Accuracy', fontsize=13, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Types of Errors
ax4 = axes[1, 1]
error_types = ['False Alarms\n(FP)', 'Missed Fraud\n(FN)']
error_vals = [FP, FN]
error_colors = ['orange', 'red']

bars = ax4.bar(error_types, error_vals, color=error_colors, alpha=0.7, 
              edgecolor='black', linewidth=2)
for bar, val in zip(bars, error_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax4.set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
ax4.set_title('Types of Mistakes Our Model Makes', fontsize=13, fontweight='bold')
ax4.set_ylim(0, max(FP, FN) * 1.3)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Does the Model Prioritize Avoiding False Alarms or Catching Fraud?
print("\n" + "="*80)
print("WHAT DOES THIS MODEL PRIORITIZE?")
print("="*80)

precision = TP / predicted_fraud

print("\nThe Big Question:")
print("Does our model care more about:")
print("A) Avoiding false alarms (not bothering customers)?")
print("B) Catching ALL fraud (even if we make mistakes)?")

print("\nANSWER: This model prioritizes AVOIDING FALSE ALARMS")

print("\nHow do we know? Let's compare the metrics:")
print(f"Precision: {precision*100:.2f}%  ← VERY HIGH!")
print(f"When we flag fraud, we're right {precision*100:.1f}% of the time")
print(f"Only {FP} false alarms out of {predicted_fraud} fraud alerts")
print(f"\nRecall: {recall*100:.2f}%  ← Good, but not perfect")
print(f"We catch {recall*100:.1f}% of fraud")
print(f"But we miss {FN} fraud cases")

print(f"\nWhat does this tell us?")
print(f"Precision ({precision*100:.1f}%) is HIGHER than Recall ({recall*100:.1f}%)")
print("This means:")
print("✅ The model is VERY careful before flagging fraud")
print("✅ It only flags transactions when it's really confident")
print("✅ Result: Very few false alarms (only {FP}!)")
print("⚠️  Trade-off: Some fraud slips through ({FN} cases)")

print("\nBusiness Strategy:")
print("This model follows a 'CUSTOMER FIRST' approach:")
print(f"\nGood news:")
print(f"Only {FP} customers experienced false declines")
print(f"{FP/(FP+TN)*100:.4f}% false alarm rate (excellent!")
print("Customers will trust the system")
print(f"Happy customers = more business")
print(f"\nTrade-off:")
print(f"{FN} fraud cases got through")
print(f"Lost about ${FN * 500:,} (assuming $500 per fraud)")
print("But kept customers happy!")

print("\nWhy is this a good strategy?")
print("1. Fraud is rare (only 1% of transactions)")
print("2. False alarms annoy MANY customers")
print("3. Happy customers = long-term profit")
print("4. Losing a few fraud cases < Losing customer trust")

print("\nAlternative strategies:")
print("\nIf we wanted to catch MORE fraud (higher recall):")
print("We'd need to flag more transactions as suspicious")
print("This would create MORE false alarms")
print("Maybe 30-50 false alarms instead of 5")
print("Would catch 2-3 more fraud cases")
print("But 30-50 angry customers!")

print("\nConclusion:")
print("This model is EXCELLENT for businesses that:")
print("• Value customer experience")
print("• Can afford to miss a small % of fraud")
print("• Want to minimize false alarms")
print("• Have fraud rates around 1%")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print("\nKey Takeaways:")
print("1. Accuracy = Overall correctness (99.85%)")
print("2. Precision = When we say fraud, are we right? (95%)")
print("3. Recall = Of all fraud, how much do we catch? (90.48%)")
print("4. F1-Score = Balance of precision and recall (92.68%)")
print("5. This model prioritizes PRECISION over RECALL")
print("6. Strategy: Customer satisfaction > catching every single fraud")
print("\nGreat job understanding classification metrics! 🎉")