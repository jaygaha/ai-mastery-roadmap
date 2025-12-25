"""
Scenario 2: Disease Screening Model

    True Positives (TP): 40 (Correctly identified positive cases)
    True Negatives (TN): 180 (Correctly identified negative cases)
    False Positives (FP): 20 (Negative cases incorrectly identified as positive)
    False Negatives (FN): 10 (Positive cases missed)

    Calculate Accuracy.
    Calculate Precision.
    Calculate Recall.
    Calculate F1-Score.
    If this disease requires immediate and costly treatment, which metric is most critical, and why?

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

print("="*80)
print("DISEASE SCREENING MODEL - CLASSIFICATION METRICS")
print("="*80)

# Data
TP = 40   # Sick people correctly identified
TN = 180  # Healthy people correctly identified  
FP = 20   # Healthy people wrongly identified as sick
FN = 10   # Sick people we missed

total = TP + TN + FP + FN
actual_positive = TP + FN  # Total sick people
actual_negative = TN + FP  # Total healthy people

# Calculate metrics
accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / actual_positive
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)

# Summary
print("\n" + "="*80)
print("METRICS SUMMARY")
print("="*80)

summary = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [f'{accuracy*100:.2f}%', f'{precision*100:.2f}%', 
              f'{recall*100:.2f}%', f'{f1_score*100:.2f}%']
})
print("\n" + summary.to_string(index=False))

# VISUALIZATIONS
print("\n" + "="*80)
print("VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Disease Screening Model - Metrics Analysis', 
             fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
ax1 = axes[0, 0]
conf_data = [[TN, FP], [FN, TP]]
sns.heatmap(conf_data, annot=True, fmt='d', cmap='RdYlGn', ax=ax1,
            xticklabels=['Predicted\nHealthy', 'Predicted\nSick'],
            yticklabels=['Actually\nHealthy', 'Actually\nSick'],
            cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

# Plot 2: Metrics Bar Chart
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
values = [accuracy*100, precision*100, recall*100, f1_score*100]
colors = ['steelblue', 'green', 'orange', 'purple']
bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax2.set_title('All Metrics', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Errors
ax3 = axes[1, 0]
error_labels = ['False\nPositives', 'False\nNegatives']
error_values = [FP, FN]
error_colors = ['orange', 'red']
bars = ax3.bar(error_labels, error_values, color=error_colors, alpha=0.7, edgecolor='black')
for bar, val in zip(bars, error_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
ax3.set_title('Model Errors', fontsize=12, fontweight='bold')
ax3.set_ylim(0, max(FP, FN) * 1.3)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Correct vs Incorrect
ax4 = axes[1, 1]
outcome_labels = ['Correct', 'Incorrect']
outcome_values = [TP + TN, FP + FN]
outcome_colors = ['green', 'red']
bars = ax4.bar(outcome_labels, outcome_values, color=outcome_colors, alpha=0.7, edgecolor='black')
for bar, val in zip(bars, outcome_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val}\n({val/total*100:.1f}%)', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')
ax4.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
ax4.set_title(f'Overall Performance', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# MOST CRITICAL METRIC
print("\n" + "="*80)
print("WHICH METRIC IS MOST CRITICAL?")
print("="*80)

print("\nContext: Disease requires immediate and costly treatment")

print("\nANSWER: RECALL is most critical!")

print("\nWhy?")
print("\n   Missing a sick person (False Negative) is DANGEROUS:")
print(f"   • {FN} sick people went home thinking they're healthy")
print("   • They don't get treatment")
print("   • Disease gets worse → could be fatal")
print("   • Cost: Patient's life at risk")

print("\n   Wrongly treating a healthy person (False Positive) is costly:")
print(f"   • {FP} healthy people get unnecessary treatment")
print("   • Wasted money on treatment")
print("   • Some side effects from treatment")
print("   • Cost: Money + discomfort")

print("\nComparison:")
print("   • Missing sick person = Life-threatening")
print("   • Treating healthy person = Expensive but not deadly")
print("   • Missing sick person is MUCH worse!")

print(f"\nOur Model Performance:")
print(f"   • Recall = {recall*100:.2f}% → We catch {recall*100:.0f}% of sick people")
print(f"   • We miss {FN} people ({FN/actual_positive*100:.0f}%)")
print(f"   • Precision = {precision*100:.2f}% → {FP} unnecessary treatments")

print("\nRecommendation:")
print("   For life-threatening diseases:")
print("   • Prioritize HIGH RECALL (catch everyone)")
print("   • Accept lower precision (some false alarms OK)")
print("   • Better to over-treat than miss someone")

print(f"\n   Example: To catch all {actual_positive} sick people (100% recall):")
print("   • Might need to flag 80-90 people as sick")
print("   • More false positives (40-50 instead of 20)")
print("   • But we save those {FN} lives we're currently missing!")

print("\nSummary:")
print("   When disease is serious → RECALL is most important")
print("   When treatment is low-risk → Can prioritize RECALL")
print("   When false alarms are costly → Balance with PRECISION")