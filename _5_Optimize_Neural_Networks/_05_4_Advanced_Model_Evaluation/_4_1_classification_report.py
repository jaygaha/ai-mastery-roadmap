from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Assume y_true and y_pred are obtained from your deep learning model
# For demonstration purposes, let's create some dummy data representing 
# a "Customer Churn" problem (0 = Stay, 1 = Leave).

np.random.seed(42)
y_true = np.random.randint(0, 2, size=100) # Actual Reality
# Simulate a model with some errors
y_pred_proba = np.random.rand(100) # Model's Confidence Scores
y_pred = (y_pred_proba > 0.5).astype(int) # Final Decision (Threshold 0.5)

# Let's make the dataset slightly imbalanced to show the effect
# Create more 'no churn' cases (class 0)
y_true[:70] = 0
y_true[70:] = 1 # 70 no churn, 30 churn

# Now, simulate predictions with some errors
# Suppose our model is good at predicting 'no churn' but struggles with 'churn'
y_pred = np.copy(y_true)
# Introduce some False Positives (predict churn when no churn)
y_pred[y_true == 0][np.random.rand(70) < 0.1] = 1 # 10% of 'no churn' are wrongly predicted as 'churn'
# Introduce some False Negatives (predict no churn when churn)
y_pred[y_true == 1][np.random.rand(30) < 0.4] = 0 # 40% of 'churn' are wrongly predicted as 'no churn'

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))