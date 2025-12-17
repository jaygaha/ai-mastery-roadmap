import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the customer churn dataset (hypothetical structure)
# This assumes 'customer_churn_data.csv' has already been preprocessed
# as per Module 2, with 'Churn' as the target variable (0 or 1)
# and other columns as features.
try:
    data = pd.read_csv('customer_churn_preprocessed.csv')
except FileNotFoundError:
    print("Error: 'customer_churn_preprocessed.csv' not found.")
    print("Please ensure you have run the data preprocessing steps from Module 2.")
    # Create dummy data for demonstration if file not found
    print("Generating dummy data for demonstration purposes...")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=0, random_state=42)
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    data['Churn'] = y

# Separate features (X) and target (y)
X = data.drop('Churn', axis=1) # Features are all columns except 'Churn'
y = data['Churn']              # 'Churn' is the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
# The 'solver' parameter specifies the algorithm to use for optimization. 'liblinear' is a good default for small datasets.
# 'random_state' ensures reproducibility.
model = LogisticRegression(solver='liblinear', random_state=42)

# Train the model on the training data
print("Training Logistic Regression model...")
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Predict probabilities for the positive class (class 1 - Churn)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Get probabilities for the '1' class

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Display classification report for more detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display some predicted probabilities
print("\nFirst 10 predicted probabilities for churn:")
print(y_pred_proba[:10])

# Access model coefficients and intercept
print("\nModel Coefficients:")
print(model.coef_)
print("\nModel Intercept:")
print(model.intercept_)