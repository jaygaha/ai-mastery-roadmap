# Model Evaluation Metrics for Regression (MAE, MSE, R2)

When you're building a machine learning model to predict numbers (like house prices, temperature, or sales), you need a way to check if your model is actually doing a good job. Unlike classification, where a model is either right or wrong (e.g., "Cat" or "Dog"), regression models predict continuous values.

So, instead of asking "Is this correct?", we ask "How close is this prediction to the actual value?".

This lesson covers the three most common ways to measure that "closeness":
1.  **MAE** (Mean Absolute Error)
2.  **MSE** (Mean Squared Error)
3.  **R-Squared** ($R^2$)

---

## 1. Mean Absolute Error (MAE)

**"The Average Miss"**

MAE is the simplest metric to understand. It answers the question: **"On average, how far off are my predictions?"**

It treats all errors equally, whether positive or negative. If you predict a house costs \$300k and it differs by \$10k, it doesn't matter if you guessed too high or too low—the error is just \$10k.

### The Formula
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

*Don't let the math scare you! It just means:*
1.  Take the difference between the actual value ($y$) and predicted value ($\hat{y}$).
2.  Make it positive (absolute value).
3.  Add them all up.
4.  Divide by the number of predictions to get the average.

### Real-World Example: Predicting House Prices
Imagine you predict prices for 3 houses:

| House | Actual Price | Predicted Price | Error (Difference) | Absolute Error |
| :--- | :--- | :--- | :--- | :--- |
| House A | \$300,000 | \$310,000 | -\$10,000 | \$10,000 |
| House B | \$450,000 | \$430,000 | +\$20,000 | \$20,000 |
| House C | \$280,000 | \$290,000 | -\$10,000 | \$10,000 |

*   **Total Miss:** \$10k + \$20k + \$10k = \$40,000
*   **Average Miss (MAE):** \$40,000 / 3 ≈ **\$13,333**

**Interpretation:** "On average, our model's price predictions are off by about \$13,333."

---

## 2. Mean Squared Error (MSE)

**"The Strict Teacher"**

MSE is similar to MAE, but with a twist: it **squares** the errors before averaging them.

Why square them? Because squaring punishes **large errors** much more than small ones.
*   An error of 2 becomes 4 ($2^2$).
*   An error of 10 becomes 100 ($10^2$).

This metric is great when being "way off" is much worse than being "slightly off".

### The Formula
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Same Example: House Prices
Using the same data as above:

| House | Absolute Error | Squared Error |
| :--- | :--- | :--- |
| House A | \$10 | $10^2 = 100$ |
| House B | \$20 | $20^2 = 400$ |
| House C | \$10 | $10^2 = 100$ |

*   **Total Squared Error:** 100 + 400 + 100 = 600
*   **Average (MSE):** 600 / 3 = **200**

**Interpretation:** The MSE is 200.
*Wait, 200 what?* Since we squared the dollars, the unit is "dollars squared" ($^2$), which doesn't make intuitive sense.

### Root Mean Squared Error (RMSE)
To fix the weird unit issue, we typically take the square root of MSE.
$$RMSE = \sqrt{200} \approx 14.14$$
Now we can say: "The model is off by about \$14,140, but heavily penalized for that one bad guess on House B."

---

## 3. R-Squared ($R^2$) Score

**"The Accuracy Score"**

While MAE and MSE measure error (lower is better), $R^2$ measures **performance** (higher is better).

It answers: **"How much better is my model than just guessing the average?"**

*   **$R^2 = 1$ (100%)**: Perfect predictions.
*   **$R^2 = 0$ (0%)**: Your model is no better than just predicting the average value for everyone.
*   **Negative $R^2$**: Your model is actually *worse* than just guessing the average!

### Interpretation Guide
*   **0.90+**: Excellent fit (Explains 90% of the variation).
*   **0.70 - 0.90**: Good fit.
*   **< 0.50**: Poor fit (Review your data or features).

---

## Comparision Cheat Sheet

| Metric | Full Name | Best For... | Interpretable? |
| :--- | :--- | :--- | :--- |
| **MAE** | Mean Absolute Error | General purpose. When outliers (rare extreme values) shouldn't ruin the score. | ✅ Yes (Direct units) |
| **MSE** | Mean Squared Error | When you hate large errors. Useful for mathematical optimization. | ❌ No (Squared units) |
| **RMSE** | Root Mean Squared Error | When large errors are bad, but you still want a readable number. | ✅ Yes (Direct units) |
| **$R^2$** | R-Squared | Comparing different models. Understanding "goodness of fit". | ✅ Yes (0 to 1 scale) |

---

## Practical Examples with Python

We can calculate these easily using `scikit-learn`.

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Setup Data
actual_salaries    = [50, 65, 70, 80, 95]  # The real values
predicted_salaries = [52, 63, 72, 78, 98]  # What our model predicted

# 2. Calculate Metrics
mae = mean_absolute_error(actual_salaries, predicted_salaries)
mse = mean_squared_error(actual_salaries, predicted_salaries)
rmse = np.sqrt(mse) # RMSE is just the square root of MSE
r2 = r2_score(actual_salaries, predicted_salaries)

# 3. Print Results
print(f"MAE:  ${mae:.2f}")   # On average, off by $2.40k
print(f"RMSE: ${rmse:.2f}")  # Larger errors penalized more
print(f"R2:   {r2:.2f}")     # Model explains ~97% of salary variation
```

---

## Exercises

1. [Exercise 1: Residential Energy Consumption Prediction](./_5_2_1_example.py)
   *   Predict energy usage and calculate the error.
2. [Exercise 2: Customer Churn Prediction (Regression Component)](./_5_2_2_example.py)
   *   Advanced: Predict Customer Lifetime Value (CLV) and evaluate using all metrics.

---

## Real-World Application: Financial Forecasting

Regression metrics aren't just for homework; they drive billion-dollar decisions in finance.

### 1. Stock Price Prediction
*   **MAE**: A trader might say, "Our model is usually off by 50 cents." This is the MAE. It helps them set stop-loss limits.
*   **RMSE**: If the model is usually good but sometimes crashes completely (predicts a gain when the stock tanks), the RMSE will be huge. This warns risk managers that the model is dangerous.

### 2. Loan Default Risk
Banks predict "Loss Given Default" (how much money they lose if you don't pay back a loan).
*   **RMSE is King**: Undestimating a massive loss is far worse than overestimating a small one. Because RMSE punishes large errors, banks prioritize this metric to avoid catastrophic losses.

---

## Conclusion

In this lesson, we moved beyond simply generating predictions to **evaluating** them. You learned that:
*   **MAE** is your "friendly neighbor" metric—easy to understand and fair.
*   **MSE/RMSE** are the "strict teachers"—they harshly penalize big mistakes.
*   **R-Squared** performs a sanity check—telling you if your model is actually better than a random guess.

**Next Up:** Now that we can measure error for *numbers*, how do we measure error for *categories*? In the next module, we will explore **Classification Metrics** like Precision, Recall, and the F1-Score.