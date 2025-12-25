"""
Scenario 1: Fraud Detection Model

    True Positives (TP): 95 (Correctly identified fraudulent transactions)
    True Negatives (TN): 9800 (Correctly identified legitimate transactions)
    False Positives (FP): 5 (Legitimate transactions incorrectly flagged as fraud)
    False Negatives (FN): 10 (Fraudulent transactions missed)

    Your Task:
    1. Calculate Accuracy.
    2. Calculate Precision.
    3. Calculate Recall.
    4. Calculate F1-Score.
    5. Interpret: Does this model prioritize avoiding false alarms or catching all fraud?
"""

import numpy as np
import pandas as pd

print("="*80)
print("FRAUD DETECTION MODEL - CLASSIFICATION METRICS")
print("="*80)

# The given values
TP = 95
TN = 9800
FP = 5
FN = 10

# TODO: Calculate the total number of predictions
total = 0 # Replace 0 with formula

# TODO: Calculate Metrics
# Accuracy = (TP + TN) / Total
accuracy = 0

# Precision = TP / (TP + FP)
precision = 0

# Recall = TP / (TP + FN)
recall = 0

# F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
f1_score = 0

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
# TODO: Write a print statement explaining which metric is higher (Precision or Recall) 
# and what that means for the business.
