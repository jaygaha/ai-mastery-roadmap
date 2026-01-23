"""
Advanced Model Evaluation: Beyond Accuracy

Accuracy alone can be misleading, especially for imbalanced datasets like churn
prediction where non-churners far outnumber churners. This script demonstrates
comprehensive evaluation techniques that reveal the true performance of your model.

Metrics Covered:
    1. Confusion Matrix: Shows exactly where your model makes mistakes
       (True Positives, True Negatives, False Positives, False Negatives)
    
    2. Precision: "Of those we predicted would churn, how many actually did?"
       High precision = fewer wasted retention offers
    
    3. Recall: "Of those who actually churned, how many did we catch?"
       High recall = fewer missed churners (often more important!)
    
    4. F1-Score: Harmonic mean of precision and recall
       Good for balanced optimization
    
    5. ROC Curve & AUC: Visualizes trade-off between true positives and false 
       positives across all thresholds. AUC summarizes overall performance.
    
    6. Precision-Recall Curve: Especially useful for imbalanced datasets

Run with: conda run -n tf_env python _6_5_advanced_evaluation.py
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Re-using the dummy data preparation and best model loading
np.random.seed(42)
num_samples = 1000
features = np.random.rand(num_samples, 10) # 10 features
target = np.random.randint(0, 2, num_samples) # Binary target (churn/no churn)

gender = np.random.choice(['Male', 'Female'], num_samples)
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples)
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], num_samples)

df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(10)])
df['gender'] = gender
df['contract'] = contract
df['internet_service'] = internet_service
df['churn'] = target

df = pd.get_dummies(df, columns=['gender', 'contract', 'internet_service'], drop_first=True)

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# For demonstration, let's load a previously saved model or redefine a trained one
# In a real scenario, you'd load 'best_churn_model.keras'
try:
    best_model = keras.models.load_model('best_churn_model.keras')
except:
    print("Could not load 'best_churn_model.keras'. Training a new model for evaluation demonstration.")
    model = keras.Sequential([
        keras.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    best_model = model


# Get predictions on the test set
y_pred_probs = best_model.predict(X_test_scaled)
# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Churn Prediction')
plt.show()

# 2. Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# 3. ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Optional: Plotting Precision-Recall Curve (useful for highly imbalanced datasets)
from sklearn.metrics import precision_recall_curve
precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_pred_probs)
plt.figure(figsize=(7, 6))
plt.plot(recall_pr, precision_pr, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()
