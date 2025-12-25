# Model Evaluation Metrics for Classification (Accuracy, Precision, Recall, F1-Score)

In the previous lessons, we built models to predict categories (like "Will this customer churn?" or "Is this email spam?"). But how do we know if our model is actually *good*?

Unlike regression, where we measure how "far off" a number is (like being \$50 off in a price prediction), classification is about being **Right** or **Wrong**. But "Right" and "Wrong" can be complicated.

This lesson will teach you the essential metrics to judge a classification model: **Accuracy, Precision, Recall, and the F1-Score**.

## The Confusion Matrix: The Foundation

Before we do any math, we use a simple table called the **Confusion Matrix** to see exactly *how* our model is getting things right (or wrong).

Imagine we are building a "Wolf Detector" for a village (The "Boy Who Cried Wolf" story).
*   **Positive Case:** There is a Wolf.
*   **Negative Case:** There is No Wolf.

The matrix has four possible outcomes:

| | **Actual Wolf** | **Actual No Wolf** |
| :--- | :--- | :--- |
| **Model Says "Wolf!"** | **True Positive (TP)**<br>*(We caught the wolf!)* | **False Positive (FP)**<br>*(False Alarm! Boy cried wolf, but no wolf.)* |
| **Model Says "Safe"** | **False Negative (FN)**<br>*(Missed it! Wolf attacks, we slept.)* | **True Negative (TN)**<br>*(All quiet, and we stayed quiet.)* |

*   **True Positives (TP)**: We predicted "Yes" and it was "Yes". (Good!)
*   **True Negatives (TN)**: We predicted "No" and it was "No". (Good!)
*   **False Positives (FP) - Type I Error**: We predicted "Yes" but it was "No". (A False Alarm)
*   **False Negatives (FN) - Type II Error**: We predicted "No" but it was "Yes". (A Miss - usually dangerous!)

---

## 1. Accuracy
*How often is the model correct overall?*

$$Accuracy = \frac{Total Correct Predictions}{Total Predictions} = \frac{TP + TN}{TP + TN + FP + FN}$$

**When to use it:** When the classes are balanced (e.g., 50% wolves, 50% sheep).
**When NOT to use it:** When classes are **imbalanced**.

> **Why Accuracy Fails:**
> Imagine a village where wolves only appear 1% of the time. If I build a "Lazy Detector" that **ALWAYS says "No Wolf"**, it will be **99% Accurate**! But it's useless because it never catches a wolf.

---

## 2. Precision
*When the model claims "Positive", how often is it right?*

$$Precision = \frac{True Positives}{True Positives + False Positives}$$

**Think:** "Quality of the positive prediction."
**Goal:** Minimize **False Alarms**.

**Use Case:** **Spam Filter.**
*   If we mark a good email as Spam (False Positive), you miss an important message. This is annoying.
*   We want high Precision so users trust the "Spam" label.

---

## 3. Recall (Sensitivity)
*Of all the actual positive cases, how many did we find?*

$$Recall = \frac{True Positives}{True Positives + False Negatives}$$

**Think:** "Quantity of positive cases found."
**Goal:** Minimize **Missed Cases**.

**Use Case:** **Cancer Detection / Wolf Detector.**
*   If we tell a sick patient they are healthy (False Negative), they might die.
*   We want high Recall to ensuring we catch *every* case, even if we raise a few false alarms.

---

## 4. F1-Score
*The Balance between Precision and Recall.*

The F1-Score is the "harmonic mean" of Precision and Recall. It penalizes extreme values.

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**When to use it:**
*   You need a balance between Precision and Recall.
*   You have an uneven class distribution (imbalanced dataset).
*   Correctly classifying "Positives" is just as important as minimizing false alarms.

---

## Real-World Comparison: Precision vs. Recall

| Metric | Focus | High Cost Error | Example |
| :--- | :--- | :--- | :--- |
| **Precision** | "Do I trust the 'Yes'?" | **False Positive** (False Alarm) | Spam filter, YouTube Recommendations |
| **Recall** | "Did I capture them all?" | **False Negative** (Missed Case) | Disease screening, Fraud detection, Safety systems |

## Practical Example: Customer Churn

Using our **Customer Churn** case study:
*   **Positive:** Customer Leaves (Churn).
*   **Negative:** Customer Stays.

*   **High Precision:** If we predict they will leave, they almost certainly will. We don't waste money offering discounts to happy customers.
*   **High Recall:** We find *everyone* who might leave. We might accidentally offer discounts to happy customers (False Positives), but we won't lose anyone (Low False Negatives).

Most companies prefer **Recall** here because losing a customer is much more expensive than the cost of a small discount.

## Exercises

We have pre-calculated confusion matrices for two different scenarios. Your job is to calculate the metrics and decide which strategy is best.

1.  [**Scenario 1: Fraud Detection**](./_6_1_exercise.py) - Calculate metrics for a bank fraud model.
2.  [**Scenario 2: Disease Screening**](./_6_2_exercise.py) - Calculate metrics for a medical test.

### Solutions
*   [Solution for Scenario 1](./_6_1_solution.py)
*   [Solution for Scenario 2](./_6_2_solution.py)

## Next Steps

Now that you know *how* to measure success, we will apply these metrics to our **Logistic Regression** model in the next lesson to see how well we can predict Customer Churn!