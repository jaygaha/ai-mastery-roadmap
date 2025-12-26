# Applying Classic ML Algorithms to the Churn Prediction Case Study

Now it's time to put everything together! You've learned about Logistic Regression, Decision Trees, and Random Forests. You've also mastered how to evaluate models. In this module, we'll apply all these skills to solve a real business problem: predicting which customers are likely to leave (churn).

## What You'll Learn

By the end of this module, you'll be able to:
- Apply three different ML algorithms to the same problem
- Compare model performance using multiple metrics
- Tune hyperparameters to improve model accuracy
- Adjust decision thresholds based on business needs
- Make data-driven recommendations for model selection

## Why This Matters

In the real world, data scientists rarely use just one algorithm. Instead, they try multiple approaches and compare results. This module teaches you that essential skill using customer churn prediction as our case study.

## Implementing Logistic Regression for Churn Prediction

**Think of it like:** A weighted scorecard that calculates the probability a customer will churn.

Logistic Regression is perfect for yes/no questions (Will this customer churn? Yes or No). It looks at all the customer features (contract type, monthly charges, tenure, etc.) and calculates a probability score between 0% and 100%.

### Quick Recap: How It Works

Logistic Regression applies the logistic sigmoid function to the output of a linear equation. This transforms the linear output, which can range from negative infinity to positive infinity, into a probability value between 0 and 1. A threshold (commonly 0.5) is then applied to these probabilities to classify instances. For example, if the predicted probability of churn is 0.7, and the threshold is 0.5, the customer is classified as churning.

### Practical Implementation Steps

1. **Data Splitting:** Divide the preprocessed dataset into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate its performance on unseen data. A common split ratio is 70-80% for training and 20-30% for testing.
2. **Model Initialization and Training:** Instantiate the `LogisticRegression` model from `sklearn.linear_model`. The model is then trained on the training data using the `fit()` method.
3. **Prediction:** Use the trained model to make predictions on the test set. This involves predicting probabilities and then converting them into binary class labels.
4. **Evaluation:** Assess the model's performance using classification metrics such as accuracy, precision, recall, and F1-score.

## Code Example: Logistic Regression

[_7_1_logistic_regression.py](./_7_1_logistic_regression.py)

## Applying Decision Tree for Churn Prediction

**Think of it like:** A flowchart of yes/no questions that leads to a prediction.

Decision Trees are super intuitive! They work like playing "20 Questions" - asking a series of questions about the customer ("Is their contract month-to-month?" "Do they have tech support?") until reaching a final prediction. The best part? You can actually see and understand the decision-making process.

### Quick Recap: How It Works

A Decision Tree makes decisions by asking a series of questions about the data's features. Each internal node represents a test on an attribute, each branch represents an outcome of the test, and each leaf node represents a class label (churn or no churn). The tree is built by recursively splitting the data based on features that provide the best separation between classes, often using metrics like Gini impurity or entropy.

### Practical Implementation Steps

1. **Data Splitting:** Divide the preprocessed dataset into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate its performance on unseen data. A common split ratio is 70-80% for training and 20-30% for testing.
2. **Model Initialization and Training:** Instantiate the `DecisionTreeClassifier` model from `sklearn.tree`. The model is then trained on the training data using the `fit()` method.
3. **Prediction:** Use the trained model to make predictions on the test set. This involves predicting probabilities and then converting them into binary class labels.
4. **Evaluation:** Assess the model's performance using classification metrics such as accuracy, precision, recall, and F1-score.

## Code Example: Decision Tree

We will build on the preprocessed data from our Customer Churn case study.

1. **Load and Prepare Data:** Use the same preprocessing pipeline we established, ensuring `X_train_processed`, `X_test_processed`, `y_train`, and `y_test` are ready.
2. **Initialize and Train Decision Tree Classifier:** Use `sklearn.tree.DecisionTreeClassifier`.
3. **Make Predictions:** Predict churn on the test set.
4. **Evaluate Model:** Use classification metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix) and visualize the tree.


[_7_2_decision_tree.py](./_7_2_decision_tree.py)

## Utilizing Ensemble Methods: Random Forests for Churn Prediction

**Think of it like:** Asking 100 experts instead of just one, then going with the majority vote.

Random Forests build many Decision Trees (typically 100+) and let them all vote on the prediction. This "wisdom of the crowd" approach is usually more accurate than a single tree. It's like getting a second (and third, and fourth...) opinion before making an important decision.

### Quick Recap: How It Works

A Random Forest constructs a multitude of Decision Trees at training time. For classification, the output of the Random Forest is the class selected by most trees (voting). Key elements include:

* **Bagging (Bootstrap Aggregating):** Each tree is trained on a different bootstrap sample (random sampling with replacement) of the training data.
* **Feature Randomness:** When splitting a node, each tree considers only a random subset of features. This decorrelates the trees, making the ensemble more robust.

### Practical Implementation Steps

1. **Data Splitting:** Continue using the same train-test split.
2. **Model Initialization and Training:** Instantiate the RandomForestClassifier from sklearn.ensemble. Important hyperparameters include n_estimators (number of trees in the forest) and max_depth. Train the model.
3. **Prediction:** Generate predictions on the test set.
4. **Evaluation:** Use the same classification metrics.


## Code Example: Random Forest

[_7_3_random_forest.py](./_7_3_random_forest.py)

## Comparing Model Performance for Churn Prediction

After training multiple models, comparing their performance using various evaluation metrics is crucial. The choice of the "best" model depends on the specific business objectives. For churn prediction, identifying potential churners is often prioritized, even if it means classifying some non-churners incorrectly. In such cases, Recall for the churn class is often a critical metric.

| Metric	| Logistic Regression	| Decision Tree	| Random Forest	| Notes
| --- | --- | --- | --- | ---
| Accuracy	| 0.795	| 0.780	| 0.810	| Overall correct predictions. Good for balanced datasets.
| Precision	| 0.650	| 0.620	| 0.700	| Of all predicted churners, how many actually churned.
| Recall	| 0.550	| 0.580	| 0.600	| Of all actual churners, how many were correctly identified.
| F1-Score	| 0.596	| 0.599	| 0.647	| Harmonic mean of precision and recall. Balances both.

_Note:_ The values in this table are illustrative and will vary based on the specific dataset and model parameters.

In a churn prediction scenario:

* A high **Recall** for the churn class means the model is good at catching actual churners, which is vital for intervention strategies.
* A high **Precision** means that when the model predicts churn, it's usually correct, minimizing wasted resources on incorrectly identified customers.
* The **F1-Score** provides a balance, especially when there's an uneven class distribution (e.g., fewer churners than non-churners).

The **Random Forest** model generally performs well due to its ensemble nature, often showing better generalization capabilities compared to a single Decision Tree. Logistic Regression provides a good baseline and interpretability through its coefficients.

## Exercises and Practice Activities

Ready to practice? These exercises will help you master model comparison and optimization:

### Exercise 1: Hyperparameter Tuning

**Goal:** Learn how changing model settings affects performance.

* For the Decision Tree model, experiment with different `max_depth` values (e.g., 3, 7, 10) and observe how accuracy, precision, and recall change. What happens if `max_depth` is too small or too large?
* For the Random Forest model, vary `n_estimators` (e.g., 50, 200, 500) and `max_features` (e.g., 'auto', 0.5, 'sqrt'). Analyze the impact on model performance.

**Solution:** [_7_4_exercise_solution.py](./_7_4_exercise_solution.py) (Exercises 1 & 2)

### Exercise 2: Feature Importance

**Goal:** Discover which customer characteristics matter most for churn prediction.

* Extract and visualize the feature importances from the Random Forest model. Identify the top 5 most important features contributing to churn prediction. How do these align with your understanding of customer behavior?

**Solution:** [_7_4_exercise_solution.py](./_7_4_exercise_solution.py) (Exercise 3)

### Exercise 3: Threshold Adjustment

**Goal:** Learn to balance precision and recall based on business priorities.

* Instead of using the default 0.5 threshold for classification, try adjusting the threshold for the Logistic Regression model. For instance, if the business prioritizes capturing more churners (higher recall), you might lower the threshold to 0.3. How does this impact precision and recall?

**Solution:** [_7_4_exercise_solution.py](./_7_4_exercise_solution.py) (Exercise 4)

### Exercise 4: Model Comparison

**Goal:** Make a data-driven recommendation for which model to deploy.

* Create a summary table comparing the final metrics (Accuracy, Precision, Recall, F1-Score) of all three models on the test set. Discuss which model you would recommend for a business aiming to minimize actual churn, and why.

**Solution:** [_7_4_exercise_solution.py](./_7_4_exercise_solution.py) (Exercise 5)

## Real-World Application

In telecommunications, churn prediction models are critical. Companies like Vodafone or AT&T constantly analyze customer data to identify subscribers at risk of leaving.

* **Logistic Regression** might be used as a first-pass model due to its speed and interpretability, allowing identification of basic churn drivers like contract expiry or recent service issues. For example, a telco might use it to quickly flag customers whose contract is ending in the next month, combined with recent high-volume support calls.
* **Decision Trees** can provide an intuitive flow for understanding specific churn paths. A financial institution, for instance, could use a Decision Tree to visualize that customers with low savings account balances and no active credit products are highly likely to close their accounts. This helps define targeted retention campaigns.
* **Random Forests** are often deployed in production for their robustness and higher accuracy. An e-commerce platform like Amazon or a streaming service like Netflix might use a Random Forest to predict which subscribers are likely to cancel their subscriptions based on viewing habits, engagement, payment history, and recent interactions. The model can identify complex patterns that individual trees might miss, such as a combination of reduced watch time, increased support queries, and a past history of switching services. This allows for proactive offers or personalized content recommendations to retain those users.

## Conclusion and Next Steps

This lesson demonstrated the practical application of Logistic Regression, Decision Trees, and Random Forests to the customer churn prediction case study. You've seen how to implement these algorithms using scikit-learn, make predictions, and evaluate their performance using key classification metrics. Understanding these classic algorithms and their strengths and weaknesses is fundamental. The Random Forest often emerges as a strong contender due to its ensemble nature, but the "best" model always depends on the specific business context and the relative importance of precision vs. recall.

In the next lesson, we will begin exploring Deep Learning, starting with the foundational concepts of neural networks, which offer a different paradigm for tackling complex prediction tasks, including churn. While these classic ML algorithms are powerful, deep learning can sometimes capture even more intricate patterns in very large or complex datasets.