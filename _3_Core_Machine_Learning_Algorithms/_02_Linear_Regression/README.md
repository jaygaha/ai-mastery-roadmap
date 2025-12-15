# Linear Regression: Theory, Assumptions, and Practical Implementation (`Scikit-Learn`)

Linear regression is often the first algorithm data scientists learn, and for good reason—it's simple, interpretable, and powerful. At its heart, it helps us predict a number (like a house price or a test score) based on other related information (like square footage or hours studied).

## Understanding the Linear Model

Imagine you're trying to draw a straight line through a cloud of data points. You want that line to be as close to all the points as possible. That's exactly what linear regression does: it finds the "best-fitting" line that describes the relationship between your inputs (independent variables) and your target (dependent variable).

### Simple Linear Regression

In simple linear regression, there is one independent variable and one dependent variable. The relationship is modeled by a straight line equation:

$y = \beta_0 + \beta_1 x + \epsilon$

Here:

* $y$ is the dependent variable (target, the value we want to predict)
* $x$ is the independent variable (feature, the input variable)
* $\beta_0$ is the y-intercept (the value of $y$ when $x=0$)
* $\beta_1$  is the coefficient for xx, representing the change in yy for a one-unit change in xx. It describes the slope of the line.
* $\epsilon$ is the error term (the difference between the observed value and the predicted value)


**Example 1: Predicting House Prices** Consider predicting house prices ($y$) based on their size in square feet ($x$). A simple linear model might be: `Price = Beta_0 + Beta_1 * Size + Error` Here, `Beta_0` could represent the base value of a house with zero size (though practically, this interpretation is limited), and `Beta_1` would represent how much the price increases for every additional square foot.

**Example 2: Advertising Spend vs. Sales** Imagine a company wants to predict product sales ($y$) based on its advertising budget ($x$) for a particular month. `Sales = Beta_0 + Beta_1 * Advertising_Budget + Error` In this scenario, `Beta_0` might represent the baseline sales achieved even with no advertising, and `Beta_1` would quantify the average increase in sales for every dollar spent on advertising.

**Hypothetical Scenario: Predicting Student Test Scores** A school wants to understand if the number of hours a student studies ($x$) influences their test scores ($y$). `Test_Score = Beta_0 + Beta_1 * Study_Hours + Error` If `Beta_1` is positive, it suggests that more study hours lead to higher test scores. `Beta_0` would be the predicted test score for a student who studied zero hours.

### Multiple Linear Regression

When there are two or more independent variables, the model extends to multiple linear regression:

$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon$

Here:

* $x_1, x_2, \ldots, x_n$ are the nn independent variables (features).
* $\beta_1, \beta_2, \ldots, \beta_n$ are the coefficients corresponding to each feature, indicating the change in yy for a one-unit change in that feature, holding all other features constant.

**Example: Predicting House Prices with Multiple Features** Extending the house price example, we might also include the number of bedrooms ($x_2$) and distance to the city center ($x_3$) as features: `Price = Beta_0 + Beta_1 * Size + Beta_2 * Bedrooms + Beta_3 * Distance_to_City + Error` `Beta_2` indicates the change in price for an additional bedroom, assuming size and distance to the city remain constant. `Beta_3` shows the price change for a unit increase in distance to the city, holding size and bedrooms constant.

### The Objective: Minimizing Residuals

How does the computer define "best-fitting"? It looks at the **residuals**—essentially, the mistakes the line makes. A residual is the vertical distance between a real data point and the line.

The goal is to find the line that makes the total size of these mistakes as small as possible. Specifically, it tries to minimize the **Sum of Squared Residuals (RSS)**. Think of it as a game of "limbo" where the line tries to get the total error bar as low as possible.

$Residual$ = $y$ $-$ $\hat{y}$

To find the "best-fitting" line, linear regression typically uses the **Ordinary Least Squares** (OLS) method. OLS aims to minimize the sum of the squared residuals (RSS):

$RSS=\sum_{i=1}^{m}(y_i-y^i)^2$

Minimizing the squared errors ensures that large errors are penalized more heavily and prevents positive and negative errors from canceling each other out. The coefficients that minimize this sum are considered the optimal parameters for the linear model.

### Key Assumptions of Linear Regression

Linear regression is powerful, but it's not magic. It relies on a few "ground rules" to work correctly. If your data breaks these rules, your model's predictions (and especially its statistical confidence) might be shaky.

Think of these as a checklist to run through before trusting your model 100%.

#### 1. Linearity

The relationship between the independent variables and the dependent variable must be linear. This means that the change in the dependent variable for a one-unit change in an independent variable is constant across the range of the independent variable.

**Implication of Violation:** If the true relationship is non-linear (e.g., quadratic), a linear model will fail to capture the underlying pattern, leading to poor predictions and biased coefficients.

**How to Check:**

- **Scatter plots:** Plot the dependent variable against each independent variable. Look for roughly linear patterns.
- **Residual plots:** Plot residuals against predicted values or independent variables. A pattern in the residual plot (e.g., a U-shape) suggests non-linearity.

**Handling Violation:**


- **Transformations:** Apply non-linear transformations to variables (e.g., logarithmic, square root) to linearize the relationship.
- **Polynomial features:** Introduce polynomial terms (e.g., $x^2, x^3$) to capture curvature. This moves into polynomial regression, a form of linear regression.
- **Non-linear models:** Use other regression techniques better suited for non-linear relationships.

#### 2. Independence of Errors (No Autocorrelation)

The errors (residuals) should be independent of each other. This means that the error for one observation should not be related to the error for another observation. This assumption is particularly important for time-series data where consecutive observations often exhibit dependence.

**Implication of Violation:** Autocorrelation (correlated errors) leads to biased standard errors, making hypothesis tests and confidence intervals unreliable. The model might appear more accurate than it actually is.

**How to Check:**

- **Durbin-Watson test:** A statistical test specifically designed to detect autocorrelation. Values near 2 suggest no autocorrelation.
- **Residual plots over time:** If data is time-series, plot residuals against time. Look for patterns (e.g., cycles, trends).

**Handling Violation:**

- **Time-series specific models:** Use models like ARIMA for time-series data.
- **Lagged variables:** Include lagged versions of the dependent or independent variables as predictors.
- **Generalized Least Squares (GLS):** A more advanced method that can account for correlated errors.

#### 3. Homoscedasticity (Constant Variance of Errors)

The variance of the errors should be constant across all levels of the independent variables. In other words, the spread of residuals should be roughly the same regardless of the predicted value or the value of any independent variable.

**Implication of Violation (Heteroscedasticity):** Heteroscedasticity means the spread of errors changes. This does not bias the coefficient estimates themselves, but it makes the OLS estimates of the standard errors inefficient and biased. This affects the reliability of hypothesis tests and confidence intervals. The model gives undue weight to observations from subsets where the error variance is larger.

**How to Check:**

- **Residual plots:** Plot residuals against predicted values $\hat{y}$ or independent variables. Look for a "fan" shape (widening or narrowing spread of residuals) or other discernible patterns.

**Handling Violation:**

- **Transformations:** Apply transformations to the dependent variable (e.g., log transformation often helps stabilize variance).
- **Weighted Least Squares (WLS):** A method that assigns different weights to observations based on the inverse of their error variance.
- **Robust standard errors:** Calculate standard errors that are robust to heteroscedasticity, providing more reliable inference even if the assumption is violated.

#### 4. Normality of Errors

The errors (residuals) should be approximately normally distributed. This assumption is less critical for the coefficient estimates themselves, especially with large sample sizes, due to the Central Limit Theorem. However, it is important for the validity of statistical inference (e.g., p-values, confidence intervals).

**Implication of Violation:** If errors are not normally distributed, confidence intervals and hypothesis tests based on OLS will not be accurate. While OLS coefficient estimates remain unbiased, their precision might be misrepresented.

**How to Check:**

- **Histogram of residuals:** Visually inspect the distribution of residuals.
- **Q-Q plot (Quantile-Quantile plot):** Compares the quantiles of the residuals to the quantiles of a normal distribution. Points should lie close to a straight line.
- **Shapiro-Wilk test, Kolmogorov-Smirnov test:** Statistical tests for normality (though sensitive to sample size).

**Handling Violation:**

- **Transformations:** Transform the dependent variable if the distribution of errors is highly skewed.
- **Non-parametric methods:** Consider alternative regression techniques that do not rely on normality assumptions.
- **Larger sample size:** For large samples, the Central Limit Theorem often allows for valid inference even with non-normal errors.

#### 5. No Multicollinearity

Independent variables should not be highly correlated with each other. Multicollinearity refers to a situation where two or more predictor variables in a multiple regression model are highly correlated.

**Implication of Violation:**

- **Unstable coefficients:** It becomes difficult to estimate the individual impact of each highly correlated independent variable on the dependent variable because their effects are intertwined. Small changes in the data can lead to large changes in the estimated coefficients.
- **Inflated standard errors:** This makes it harder to reject the null hypothesis that a coefficient is zero, even if it is truly significant.
- **Reduced interpretability:** It is challenging to interpret the individual coefficients meaningfully.

**How to Check:**

- **Correlation matrix:** Calculate the correlation between all pairs of independent variables. High absolute correlation coefficients (e.g., > 0.7 or 0.8) indicate potential multicollinearity.
- **Variance Inflation Factor (VIF):** A more robust measure. VIF quantifies how much the variance of an estimated regression coefficient is inflated due to multicollinearity. A VIF value greater than 5 or 10 is often considered problematic.

**Handling Violation:**

- **Remove one of the correlated variables:** If two variables are highly correlated, removing one might be the simplest solution.
- **Combine correlated variables:** Create an index or composite variable from the highly correlated predictors.
- **Dimensionality reduction:** Techniques like Principal Component Analysis (PCA) can transform correlated variables into a smaller set of uncorrelated components.
- **Regularization techniques:** Methods like Ridge Regression (covered later in the course) are designed to handle multicollinearity by adding a penalty term to the OLS cost function.

### Practical Implementation with Scikit-Learn

Scikit-learn is a powerful Python library for machine learning. It provides a simple and consistent API for implementing various algorithms, including linear regression.

#### Setting up the Environment and Data

First, ensure you have `numpy`, `pandas`, `matplotlib`, and `scikit-learn` installed.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

We wil use a synthetic dataset for demostrating the implementation.

```python
# Generate synthetic data
np.random.seed(42) # for reproducibility

# Feature 1: Study Hours
X_study_hours = np.random.rand(100, 1) * 10 + 2 # 2 to 12 hours
# Feature 2: Previous Exam Score (correlated with study hours)
X_prev_score = X_study_hours * 5 + np.random.randn(100, 1) * 10 + 50 # 50 to 120
# Feature 3: Aptitude Score (less correlated)
X_aptitude = np.random.rand(100, 1) * 50 + 50 # 50 to 100

# Target: Final Exam Score
# Final_Score = 30 + 5*Study_Hours + 0.5*Previous_Exam_Score + 0.2*Aptitude_Score + noise
y_final_score = (30 + 5 * X_study_hours + 0.5 * X_prev_score + 0.2 * X_aptitude +
                 np.random.randn(100, 1) * 15).flatten() # Added some noise and flattened

# Combine features into a DataFrame
data = pd.DataFrame({
    'Study_Hours': X_study_hours.flatten(),
    'Previous_Exam_Score': X_prev_score.flatten(),
    'Aptitude_Score': X_aptitude.flatten(),
    'Final_Exam_Score': y_final_score
})

print("First 5 rows of the synthetic dataset:")
print(data.head())
```

#### Data Splitting

Before training any model, it is crucial to split the data into training and testing sets. This allows us to evaluate the model's performance on unseen data.

```python
# Define features (X) and target (y)
X = data[['Study_Hours', 'Previous_Exam_Score', 'Aptitude_Score']]
y = data['Final_Exam_Score']

# Split data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing
# random_state for reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
```

#### Model Training

Scikit-learn's `LinearRegression` model fits a linear model with coefficients $w=(w_1,\ldots,w_p)$ to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

```python
# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

print("\nModel training complete.")
```

#### Inspecting Model Coefficients

After training, you can inspect the learned coefficients ($w=(w_0,w_1,…,w_p)$).

```python
# The intercept (Beta_0)
print(f"Intercept (Beta_0): {model.intercept_:.2f}")

# The coefficients for each feature (Beta_1, Beta_2, ..., Beta_n)
print("Coefficients (Beta_n for each feature):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"- {feature}: {coef:.2f}")
```

**Interpretation of Coefficients:**

* A positive coefficient means that as the value of that feature increases, the target variable tends to increase.
* A negative coefficient means that as the value of that feature increases, the target variable tends to decrease.
* The magnitude of the coefficient indicates the strength of the relationship (assuming features are on a comparable scale).

For example, a coefficient of `5.01` for `Study_Hours` suggests that for every additional hour of study, the final exam score is predicted to increase by approximately 5.01 points, holding `Previous_Exam_Score` and `Aptitude_Score` constant.

#### Making Predictions

Once the model is trained, it can be used to make predictions on new, unseen data (our test set).

```python
# Make predictions on the test set
y_pred = model.predict(X_test)

print("\nFirst 5 actual test scores:", y_test.head().tolist())
print("First 5 predicted test scores:", y_pred[:5].tolist())
```

#### Model Evaluation

To assess how well our linear regression model performs, we use evaluation metrics. For regression tasks, common metrics include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2R2). These will be covered in detail in a future lesson, but we will briefly use them here.

```python
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.2f}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2:.2f}")
```

**Brief Explanation of Metrics:**

* **MSE/RMSE:** Measure the average squared/root squared difference between predicted and actual values. Lower values indicate a better fit.
* **R-squared:** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. Values range from 0 to 1, with higher values indicating a better fit. An $R^2$ of 0.85 means 85% of the variance in final exam scores can be explained by our features.

#### Visualizing the Results

Visualizing the predictions can provide an intuitive understanding of model performance. For multiple linear regression, plotting all features against the target simultaneously is difficult, but we can plot actual vs. predicted values.

```python
# Plotting actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Diagonal line
plt.xlabel("Actual Final Exam Score")
plt.ylabel("Predicted Final Exam Score")
plt.title("Actual vs. Predicted Final Exam Scores")
plt.grid(True)
plt.show()

# Plotting residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel("Predicted Final Exam Score")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.grid(True)
plt.show()
```

The residual plot helps to visually check the homoscedasticity assumption. If the points are randomly scattered around the horizontal line at zero, it suggests constant variance of errors. Any patterns (e.g., a fanning out or a curved shape) would indicate heteroscedasticity or non-linearity. In our synthetic data, the residuals appear randomly scattered, which is expected.

### Exercises

* [Simple Linear Regression Exercise](./_2_2_1_exercise.py)
* [Investigating an Assumption Violation (Hypothetical)](./_2_2_2_exercise.py)
* [Customer Churn Prediction Case Study - Initial Exploration](./_2_2_3_exercise.py)

## Summary and Next Steps

This lesson introduced the fundamental theory of linear regression, differentiating between simple and multiple linear regression, and explaining the objective of minimizing the sum of squared residuals using OLS. A crucial part of using linear regression effectively is understanding its underlying assumptions: linearity, independence of errors, homoscedasticity, normality of errors, and no multicollinearity. Violating these assumptions can lead to an unreliable model and misleading interpretations. We also performed a practical implementation using Scikit-learn, covering data splitting, model training, coefficient inspection, prediction, and basic evaluation.

In the next lesson, we will delve into **Logistic Regression**, which extends the concept of linear models to classification problems, specifically addressing the limitations of linear regression when the target variable is categorical, as alluded to in the churn case study exercise. We will also explore how it uses a probabilistic approach to classification.