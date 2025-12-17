# Logistic Regression: Theory, Probabilistic Classification, and Practical Implementation

Logistic regression is the go-to algorithm when you need to answer "Yes" or "No". Will this customer churn? Is this email spam? Is this tumor malignant?

Unlike linear regression, which gives you a raw number (like "price is 500k"), logistic regression gives you a **probability** (like "85% chance of churn"). It does this by wrapping the output of a linear equation in a special "S-shaped" curve called the **sigmoid function**.

## The Logistic Function (Sigmoid Function)

Think of the Linear Regression line as a ramp that goes up to infinity. Probabilities, however, must stay between 0 and 1. The Sigmoid function is the mathematical tool that squashes that infinite ramp into a nice S-curve bound between 0 (0%) and 1 (100%).

The mathematical formula for the logistic function is:

$P(Y=1∣X)=\frac{1}{1+e^{-(b_0+b_1x_1+...+b_nx_n)}}$

Where:

* $P(Y=1∣X)$ is the probability that the dependent variable $Y$ is 1, given the features $X$.
* $e$ is the base of the natural logarithm (approximately 2.71828).
* $b_0$ is the intercept term.
* $b_1,...,b_n$ are the coefficients for the independent variables $x_1,...,x_n$.
* The term $(b_0+b_1x_1+...+b_nx_n)$ is the linear combination of input features, often denoted as $z$.

This sigmoid function transforms the linear output $z$ into a probability. As $z$ approaches positive infinity, the sigmoid output approaches 1. As $z$ approaches negative infinity, the sigmoid output approaches 0. When $z$ is 0, the sigmoid output is 0.5.

### Example: Credit Card Default Prediction

Consider a banking scenario where we want to predict if a customer will default on their credit card payment based on their income. If we used a linear model, the output could be any real number, which doesn't make sense for a probability. The sigmoid function converts this linear output into a probability between 0 and 1. For instance, if a customer's income and other factors lead to a linear score ($z$) of 2, the sigmoid function would output $1/(1+e^(-2))≈0.88$. This suggests an 88% probability of default. If another customer's factors result in a $z$ of -3, the sigmoid output would be $1/(1+e^3)≈0.05$, indicating a 5% probability of default.


### Hypothetical Scenario: Alien Species Classification

Imagine an alien planet where scientists want to classify creatures into two categories: "friendly" (0) or "hostile" (1) based on their body mass. A logistic regression model would take the creature's body mass (and potentially other features) and calculate a linear score. This score is then passed through the sigmoid function to output a probability. If a creature has a body mass that results in a high linear score, say $z=5$, the sigmoid output would be near 1, indicating a high probability of being hostile. If a creature's mass results in a low linear score, say $z=−4$, the sigmoid output would be near 0, indicating a low probability of being hostile.

## Probabilistic Classification and Decision Boundary

Logistic regression spits out a probability (e.g., 0.85). But eventually, you need to make a hard decision: Is it Yes or No?

This is where the **Decision Boundary** or **Threshold** comes in. The standard rule is:
- If Probability ≥ 0.5 $\rightarrow$ Predict 1 (Yes/Positive)
- If Probability < 0.5 $\rightarrow$ Predict 0 (No/Negative)

The threshold of 0.5 is a common default, but it can be adjusted based on the specific problem and the relative costs of false positives versus false negatives.

### Example: Customer Churn Prediction

In the customer churn prediction case study, we aim to predict whether a customer will churn (class 1) or not churn (class 0). A logistic regression model will output a probability of churn for each customer. For example, a customer with a predicted churn probability of 0.75 would be classified as "churn" if the threshold is 0.5. A customer with a predicted probability of 0.30 would be classified as "no churn". If missing a churner is much more costly than incorrectly predicting a non-churner as a churner (a false positive), the threshold might be lowered to, say, 0.3. This means any customer with a churn probability of 0.3 or higher would be flagged as a churn risk, increasing the recall for churners at the expense of potentially more false positives.

## Interpreting Coefficients

The behavior of features in logistic regression is similar to linear regression, but with a twist.

- **Positive Coefficient:** As this feature increases, the *probability* of the event (Y=1) goes **up** (S-curve shifts right).
- **Negative Coefficient:** As this feature increases, the *probability* of the event goes **down** (S-curve shifts left).

*Note: Technically, they change the "log-odds" linearly, but for general intuition, thinking about directionality (up or down) is sufficient.*

### Example: Disease Risk Assessment

Consider a model predicting the probability of developing a certain disease (1) versus not developing it (0) based on factors like age and blood pressure. If the coefficient for "age" is positive, it means that as a person's age increases, the log-odds of developing the disease increase. This implies a higher probability of disease for older individuals, all else being equal. If the coefficient for "exercise frequency" (e.g., hours per week) is negative, it means that as exercise frequency increases, the log-odds of developing the disease decrease, suggesting a lower probability of disease for more active individuals.

## Practical Implementation with `Scikit-learn`

Implementing logistic regression in Python using the `scikit-learn` library is straightforward. The process involves importing the `LogisticRegression` class, creating an instance of the model, training it on the data, and then making predictions.

```python
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
```

**Explanation of code:**

* `pd.read_csv('customer_churn_preprocessed.csv')`: This line loads the preprocessed customer churn dataset. In Module 2, you learned about preparing data, including handling missing values, encoding categorical variables, and scaling numerical features. This `customer_churn_preprocessed.csv` file represents the output of those steps, ready for model training. The `try-except` block handles cases where the file might not exist, providing dummy data for demonstration.
* `X = data.drop('Churn', axis=1)`: Separates the features (independent variables) from the target variable. `axis=1` indicates that 'Churn' is a column.
* `y = data['Churn']`: Selects the 'Churn' column as the target variable (dependent variable).
* `train_test_split(X, y, test_size=0.2, random_state=42)`: Splits the dataset into training and testing sets. `test_size=0.2` means 20% of the data will be used for testing, and `random_state=42` ensures the split is reproducible. This prevents data leakage where the model sees test data during training.
* `LogisticRegression(solver='liblinear', random_state=42)`: Initializes the logistic regression model.

    * `solver='liblinear'` is a good choice for smaller datasets and handles both L1 and L2 regularization. Other solvers like 'lbfgs', 'newton-cg', 'sag', 'saga' exist and are more suitable for larger datasets or specific regularization types.
    * `random_state` ensures that the internal random processes of the solver are consistent, making results reproducible.
* `model.fit(X_train, y_train)`: Trains the logistic regression model using the training features (`X_train`) and their corresponding target labels (`y_train`). During this step, the model learns the optimal coefficients (b0,b1,…,bnb0​,b1​,…,bn​) that best fit the training data.
* `model.predict(X_test)`: Uses the trained model to predict the class labels (0 or 1) for the unseen test data (`X_test`). By default, it applies the 0.5 probability threshold.
* `model.predict_proba(X_test)[:, 1]`: This is a crucial method for logistic regression. It returns the probability estimates for each class. [:, 1] specifically extracts the probabilities for the positive class (class 1, i.e., churn).
* `accuracy_score(y_test, y_pred)`: Calculates the overall accuracy of the model on the test set. Accuracy is one of many evaluation metrics, and its suitability depends on the dataset's class balance.
* `classification_report(y_test, y_pred)`: Provides a more comprehensive report including precision, recall, and F1-score for each class, which are essential for classification tasks, especially with imbalanced datasets. These metrics will be covered in detail in an upcoming lesson.
* `model.coef_` and `model.intercept_`: These attributes allow access to the learned coefficients and intercept of the logistic regression model, which represent the bibi​ values from the logistic function equation.

## Exercises
Solutions can be found in: [`_3_2_exercise_threshold.py`](_3_2_exercise_threshold.py)

1. **Threshold Adjustment Experiment:**

    * Using the `y_pred_proba` array generated in the practical implementation, experiment with different probability thresholds (e.g., 0.3, 0.7) to classify customers as churn or no-churn.
    * For each new threshold, calculate and print the `accuracy_score` and `classification_report`.
    * Observe how changing the threshold impacts the precision and recall for the 'churn' class (class 1). Which metric tends to improve or degrade more significantly with different thresholds?
    * Hint: You can use a list comprehension or `np.where` to convert probabilities to class labels based on your chosen threshold.

        ```python
        import numpy as np
        # Assume y_pred_proba and y_test are available from the previous code block

        # Experiment with a lower threshold (e.g., 0.3)
        custom_threshold = 0.3
        y_pred_custom_threshold = (y_pred_proba >= custom_threshold).astype(int)

        print(f"\n--- Evaluation with Threshold = {custom_threshold} ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_custom_threshold):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_custom_threshold))

        # Experiment with a higher threshold (e.g., 0.7)
        custom_threshold = 0.7
        y_pred_custom_threshold = (y_pred_proba >= custom_threshold).astype(int)

        print(f"\n--- Evaluation with Threshold = {custom_threshold} ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_custom_threshold):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_custom_threshold))
        ```

2. **Feature Impact Analysis:**

    * Using the `model.coef_` attribute, analyze the magnitude and sign of the coefficients for each feature in your (dummy or actual) customer churn dataset.
    * Identify which features have the strongest positive correlation with churn probability and which have the strongest negative correlation.
    * Discuss how these findings align with potential real-world insights for customer churn. For the dummy data, you can just label them as `feature_0`, `feature_1`, etc.
        ```python
        # Assume model and X are available from the previous code block

        # Get feature names
        feature_names = X.columns if not isinstance(X, np.ndarray) else [f'feature_{i}' for i in range(X.shape[1])]

        # Create a DataFrame to easily view coefficients
        coefficients_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0] # model.coef_ is usually a 2D array for binary classification
        })

        # Sort by absolute coefficient value to see the strongest impacts
        coefficients_df['Abs_Coefficient'] = np.abs(coefficients_df['Coefficient'])
        coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)

        print("\nFeature Coefficients (Sorted by Absolute Value):")
        print(coefficients_df)

        print("\nInterpretation Notes:")
        print("- A positive coefficient indicates that an increase in the feature value leads to an increase in the log-odds of churn (higher churn probability).")
        print("- A negative coefficient indicates that an increase in the feature value leads to a decrease in the log-odds of churn (lower churn probability).")
        print("- The magnitude of the coefficient indicates the strength of this relationship in terms of log-odds.")
        ```

## Real-World Application

Logistic regression finds widespread use across various industries due to its interpretability and effectiveness for binary classification problems.

### Medical Diagnostics

In medicine, logistic regression is commonly used to predict the presence or absence of a disease based on patient symptoms, medical history, and test results. For example, a model might predict the probability of a patient having diabetes given their age, BMI, blood sugar levels, and family history. This helps doctors prioritize further tests or interventions. Another application is predicting the risk of heart attack based on cholesterol levels, blood pressure, smoking habits, and age, providing a probability score that can inform preventative care.

### Marketing and Sales

Businesses use logistic regression to predict customer behavior. For instance, it can predict whether a customer will click on an advertisement based on their demographics, browsing history, and past interactions. This helps in optimizing ad placement and targeting. Another key application is predicting whether a potential lead will convert into a paying customer, allowing sales teams to focus on high-probability leads. This aligns directly with our churn prediction case study, where we identify customers at risk of leaving to enable targeted retention efforts.

### Fraud Detection

Financial institutions employ logistic regression to detect fraudulent transactions. By analyzing features such as transaction amount, location, frequency, and time of day, a model can assign a probability that a given transaction is fraudulent. Transactions with a high probability score can then be flagged for manual review, significantly reducing financial losses. This helps in real-time decision-making for credit card companies and banks.

## Conclusion

Logistic regression provides a powerful yet interpretable method for binary classification by modeling the probability of an event using the sigmoid function. You've explored its theoretical underpinnings, understood how it performs probabilistic classification, and implemented it using Scikit-learn on a (simulated) customer churn dataset. You've also seen how to interpret its coefficients and adjust decision thresholds.

While logistic regression is a robust and widely used algorithm, it makes certain assumptions and may not always capture complex, non-linear relationships in data. Upcoming lessons will delve into more sophisticated classification algorithms like Decision Trees and Ensemble Methods, which can model these more complex patterns, and introduce crucial evaluation metrics beyond simple accuracy to provide a more nuanced understanding of model performance. This foundation in logistic regression is critical for understanding these more advanced techniques.

    