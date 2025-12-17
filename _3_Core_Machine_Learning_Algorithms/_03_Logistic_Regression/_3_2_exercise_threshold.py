
"""
Exercise: Logistic Regression exploration
1. Algorithm Implementation
2. Threshold Adjustment Experiment
3. Feature Impact Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification

# Set plot style
sns.set_style("whitegrid")

print("*"*80)
print("LOGISTIC REGRESSION: THRESHOLDS AND COEFFICIENTS")
print("*"*80)

"""
STEP 1: LOAD AND TRAIN MODEL
"""

print("\n" + "="*80)
print("STEP 1: DATA LOADING AND MODEL TRAINING")
print("="*80)

# Load data (handling missing file with dummy data)
try:
    data = pd.read_csv('customer_churn_preprocessed.csv')
    print("Dataset loaded successfully.")
    # Assuming standard preprocessing, 'Churn' is 0/1. If not, map it.
    if data['Churn'].dtype == 'object':
         data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})
except FileNotFoundError:
    print("Warning: 'customer_churn_preprocessed.csv' not found. Using synthetic data.")
    X_syn, y_syn = make_classification(n_samples=1000, n_features=10, 
                                     n_informative=5, n_redundant=2, 
                                     random_state=42, flip_y=0.1) # flip_y adds noise
    data = pd.DataFrame(X_syn, columns=[f'feature_{i}' for i in range(10)])
    data['Churn'] = y_syn

# Split features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Select numeric features for simplicity if dataset has mixed types and wasn't fully preprocessed
# In a real pipeline, we'd handle all columns.
if 'customerID' in X.columns:
    X = X.drop('customerID', axis=1)
X = X.select_dtypes(include=[np.number])

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)
print("\nModel trained successfully.")

# Get probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

"""
STEP 2: EXERCISE - THRESHOLD ADJUSTMENT
"""

print("\n" + "="*80)
print("STEP 2: THRESHOLD ADJUSTMENT EXPERIMENT")
print("="*80)

print("\nInstruction: Observe how Precision and Recall for Class 1 (Churn) change.")

thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:
    print(f"\n--- Decision Threshold: {threshold} ---")
    
    # Apply threshold
    y_pred_custom = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred_custom)
    report = classification_report(y_test, y_pred_custom, output_dict=True)
    churn_metrics = report['1'] # Class 1 metrics
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (Churn): {churn_metrics['precision']:.4f}")
    print(f"Recall (Churn):    {churn_metrics['recall']:.4f}")
    print(f"F1-Score (Churn):  {churn_metrics['f1-score']:.4f}")
    
    # Confusion Matrix Visualization (Text based)
    cm = confusion_matrix(y_test, y_pred_custom)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Predicted Churners: {sum(y_pred_custom)} / {len(y_test)}")

print("\nInsight:")
print("- LOW Threshold (0.3): Catches more churners (High Recall), but more false alarms (Low Precision).")
print("- HIGH Threshold (0.7): Very sure when it predicts churn (High Precision), but misses many (Low Recall).")

"""
STEP 3: EXERCISE - FEATURE IMPACT ANALYSIS
"""

print("\n" + "="*80)
print("STEP 3: FEATURE IMPACT ANALYSIS (COEFFICIENTS)")
print("="*80)

# Create coefficients DataFrame
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

# Add 'Impact_Direction' for readability
coef_df['Impact'] = coef_df['Coefficient'].apply(lambda x: 'Increases Risk' if x > 0 else 'Reduces Risk')
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()

# Sort by impact strength
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

print("\nTop 10 Most Influential Features:")
print(coef_df[['Feature', 'Coefficient', 'Impact']].head(10).to_string(index=False))

print("\nInterpretation:")
print(f"• The top feature '{coef_df.iloc[0]['Feature']}' has the strongest influence.")
if coef_df.iloc[0]['Coefficient'] > 0:
    print("• Because it's positive, higher values in this feature significantly INCREASE the probability of churn.")
else:
    print("• Because it's negative, higher values in this feature significantly REDUCE the probability of churn.")

print("\n" + "*"*80)
print("ANALYSIS COMPLETE")
print("*"*80)
