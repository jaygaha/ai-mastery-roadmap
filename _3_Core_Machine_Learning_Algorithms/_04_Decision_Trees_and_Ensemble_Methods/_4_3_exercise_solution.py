
"""
Exercise Solutions: Decision Trees and Random Forests
1. Decision Tree Hyperparameter Tuning (Max Depth)
2. Random Forest Hyperparameter Tuning (N Estimators)
3. Handling Imbalanced Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# Data Loading (Robust)
def load_data():
    try:
        df = pd.read_csv('../_03_Logistic_Regression/customer_churn_preprocessed.csv')
    except FileNotFoundError:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=15, n_informative=8, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
        df['Churn'] = y
    
    if 'Churn' in df.columns and df['Churn'].dtype == 'object':
         le = LabelEncoder()
         df['Churn'] = le.fit_transform(df['Churn'])
    return df

df = load_data()
if 'customerID' in df.columns: df = df.drop('customerID', axis=1)
X = df.drop('Churn', axis=1)
y = df['Churn']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("*"*80)
print("EXERCISE 1: DECISION TREE HYPERPARAMETER TUNING (MAX DEPTH)")
print("*"*80)

depths = [3, 7, 10, None]
results_dt = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results_dt.append({'Depth': str(depth), 'Accuracy': acc, 'F1': f1})
    print(f"Max Depth: {str(depth):<5} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

best_depth = max(results_dt, key=lambda x: x['F1'])
print(f"\n-> Best Depth based on F1: {best_depth['Depth']}")

print("\n" + "*"*80)
print("EXERCISE 2: RANDOM FOREST TUNING vs DECISION TREE")
print("*"*80)

estimators_list = [50, 200, 500]
results_rf = []

for n in estimators_list:
    rf = RandomForestClassifier(n_estimators=n, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results_rf.append({'N_Est': n, 'Accuracy': acc, 'F1': f1})
    print(f"N Estimators: {n:<3} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

print("\nComparison:")
best_rf = max(results_rf, key=lambda x: x['F1'])
print(f"Best Decision Tree F1:   {best_depth['F1']:.4f}")
print(f"Best Random Forest F1:   {best_rf['F1']:.4f}")

if best_rf['F1'] > best_depth['F1']:
    print("-> Random Forest outperformed the single Decision Tree.")
else:
    print("-> Decision Tree performed similarly or better (check for overfitting in training if DT is very high).")


print("\n" + "*"*80)
print("EXERCISE 3: HANDLING IMBALANCED DATA")
print("*"*80)

print(f"Class Distribution: {y.value_counts(normalize=True).to_dict()}")

# Train vanilla Random Forest
rf_vanilla = RandomForestClassifier(random_state=42)
rf_vanilla.fit(X_train, y_train)
y_pred_vanilla = rf_vanilla.predict(X_test)
report_vanilla = classification_report(y_test, y_pred_vanilla, output_dict=True)

# Train balanced Random Forest
rf_balanced = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)
report_balanced = classification_report(y_test, y_pred_balanced, output_dict=True)

print("\n--- Vanilla Random Forest (Minority Class) ---")
print(f"Precision: {report_vanilla['1']['precision']:.4f}")
print(f"Recall:    {report_vanilla['1']['recall']:.4f}")
print(f"F1:        {report_vanilla['1']['f1-score']:.4f}")

print("\n--- Balanced Random Forest (Minority Class) ---")
print(f"Precision: {report_balanced['1']['precision']:.4f}")
print(f"Recall:    {report_balanced['1']['recall']:.4f}")
print(f"F1:        {report_balanced['1']['f1-score']:.4f}")

print("\nObservation:")
print("Using 'class_weight=balanced' usually increases Recall for the minority class,")
print("often at the cost of some Precision. This is desirable in churn prediction")
print("where missing a churner (False Negative) is costly.")
