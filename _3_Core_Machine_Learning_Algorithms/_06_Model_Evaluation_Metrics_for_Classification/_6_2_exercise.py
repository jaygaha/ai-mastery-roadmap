"""
Scenario 2: Disease Screening Model

    True Positives (TP): 40 (Correctly identified positive cases)
    True Negatives (TN): 180 (Correctly identified negative cases)
    False Positives (FP): 20 (Negative cases incorrectly identified as positive)
    False Negatives (FN): 10 (Positive cases missed)

    Your Task:
    1. Calculate Accuracy, Precision, Recall, and F1-Score.
    2. Critical Thinking: If this disease requires immediate and costly treatment, which metric is most critical?
"""

import numpy as np
import pandas as pd

print("="*80)
print("DISEASE SCREENING MODEL - CLASSIFICATION METRICS")
print("="*80)

# Data
TP = 40
TN = 180
FP = 20
FN = 10

# TODO: Calculate Metrics
accuracy = 0
precision = 0
recall = 0
f1_score = 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

print("\n" + "="*80)
print("CRITICAL THINKING")
print("="*80)
# TODO: Which metric matters most here? Why? 
# Hint: Is it worse to tell a healthy person they are sick, or to tell a sick person they are healthy?
