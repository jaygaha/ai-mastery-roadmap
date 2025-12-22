
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Robust Data Loading Function
def load_data():
    try:
        df = pd.read_csv('customer_churn_preprocessed.csv')
        print("Loaded local dataset.")
    except FileNotFoundError:
        try:
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

    if 'Churn' in df.columns and df['Churn'].dtype == 'object':
         le = LabelEncoder()
         df['Churn'] = le.fit_transform(df['Churn'])
    
    return df

df = load_data()

# Separate features (X) and target (y)
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

X = df.drop('Churn', axis=1)
y = df['Churn']
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
# n_estimators=100 is standard
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the model
print("\nRandom Forest Classifier Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Get feature importances from the Random Forest
print("\nRandom Forest Feature Importances:")
feature_importances_rf = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances_rf.head(10))