
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Robust Data Loading Function
def load_data():
    try:
        # Try local file
        df = pd.read_csv('customer_churn_preprocessed.csv')
        print("Loaded local dataset.")
    except FileNotFoundError:
        try:
            # Try sibling directory
            df = pd.read_csv('../_03_Logistic_Regression/customer_churn_preprocessed.csv')
            print("Loaded dataset from sibling directory.")
        except FileNotFoundError:
            print("Dataset not found. Generating synthetic data.")
            from sklearn.datasets import make_classification
            X_syn, y_syn = make_classification(n_samples=1000, n_features=10, 
                                             n_informative=5, n_redundant=2, 
                                             random_state=42)
            df = pd.DataFrame(X_syn, columns=[f'feature_{i}' for i in range(10)])
            df['Churn'] = y_syn
    
    # Preprocessing check
    if 'Churn' in df.columns and df['Churn'].dtype == 'object':
         le = LabelEncoder()
         df['Churn'] = le.fit_transform(df['Churn'])
         print("Encoded target variable 'Churn'.")
    
    return df

df = load_data()

# Separate features (X) and target (y)
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1) # Drop ID if present

X = df.drop('Churn', axis=1)
y = df['Churn']

# Handle categorical features if any (simple encoding for demo)
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
# max_depth=5 is a good starting point to prevent overfitting
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the model on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the model
print("\nDecision Tree Classifier Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Feature Importances
print("\nDecision Tree Feature Importances:")
if hasattr(dt_classifier, 'feature_importances_'):
    importances = pd.Series(dt_classifier.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False).head(10))

# Visualization (Optional - only if non-dummy data usually)
# For dummy data, features are named 'feature_0', etc.
try:
    plt.figure(figsize=(20,10))
    tree.plot_tree(dt_classifier, 
                   feature_names=X.columns, 
                   class_names=['No Churn', 'Churn'], 
                   filled=True, 
                   rounded=True, 
                   fontsize=10,
                   max_depth=3) # Limit depth for visualization
    plt.title("Decision Tree Visualization (Top 3 Levels)")
    # plt.show() # Commented out to prevent blocking in automation
    print("\nTree visualization created (not shown).")
except Exception as e:
    print(f"Skipping visualization: {e}")