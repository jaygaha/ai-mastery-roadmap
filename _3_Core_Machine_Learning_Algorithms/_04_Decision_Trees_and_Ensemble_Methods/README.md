# Decision Trees and Ensemble Methods (Random Forests): Theory and Implementation

Think of a **Decision Tree** like playing a game of "20 Questions". You start with a broad pool of possibilities and narrow it down by asking Yes/No questions.
- "Is it an animal?" (Yes) -> Go down the 'Animal' branch.
- "Does it fly?" (No) -> Go down the 'Walking' branch.

Eventually, you reach a final answer (a "leaf").

**Ensemble Methods**, specifically **Random Forests**, take this idea further. Instead of trusting just one person playing 20 Questions (who might be biased or make a mistake), you ask 100 people to play.
- If 90 people say "It's a Penguin" and 10 say "It's a Puffin", the "Forest" votes for "Penguin".
- This "wisdom of the crowd" approach usually gives much better results than any single tree.

## Decision Trees: Theory and Mechanics

A Decision Tree learns to ask the *best* questions to split your data into pure groups. It starts at the top (Root Node) and keeps splitting until it can't distinguish the data any further (Leaf Node).

### Tree Construction: Splitting Criteria

The construction of a decision tree involves recursively splitting the data. At each node, the algorithm chooses the best feature and split point to divide the data such that the resulting child nodes are as "pure" as possible. Purity refers to the homogeneity of the target variable within a node.

For classification tasks, common splitting criteria include:

- **Gini Impurity**: Measures the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled according to the distribution of labels in the subset. A Gini impurity of 0 indicates perfect purity (all elements belong to the same class). It is calculated as $G=1-\sum_{i=1}^{C}p_i^2$ where $C$ is the number of classes and $p_i$ is the proportion of observations belonging to class $i$ in the node. When splitting a node, the algorithm seeks to minimize the weighted average Gini impurity of the child nodes.

    Example: Consider a node with 10 instances: 6 'churn' and 4 'no churn'. The Gini impurity is $1-(6/10)^2-(4/10)^2=1-0.36-0.16=0.48$. If we split this node into two child nodes, say one with 5 'churn' and 1 'no churn' (Gini = $1-(5/6)^2-(1/6)^2=0.278$) and another with 1 'churn' and 3 'no churn' (Gini = $1-(1/4)^2-(3/4)^2=0.375$), the algorithm evaluates which split minimizes the weighted average of these child Gini values.

    *Hypothetical Scenario* : Imagine a medical diagnosis tree. A node represents 'patients with fever'. If splitting by 'sore throat' results in one branch being 'almost all flu cases' and another being 'very few flu cases', this split is preferred due to high purity in the child nodes regarding flu diagnosis.

- **Information Gain (using Entropy)**: Entropy measures the uncertainty or randomness of the data. A higher entropy indicates more mixed classes, while a lower entropy (closer to 0) indicates higher purity. Information Gain is the reduction in entropy achieved by splitting the data on a particular feature. The feature that yields the highest Information Gain is chosen for the split. Entropy is calculated as $H=\sum_{i=1}^{C}p_i\log_2(p_i)$ where $C$ is the number of classes and $p_i$ is the proportion of observations belonging to class $i$ in the node. Information Gain is $IG(S,A)=H(S)-\sum_{v \in Values(A)}\frac{|S_v|}{|S|}H(S_v)$ where $S$ is the set of examples, $A$ is the attribute, $Values(A)$ are the possible values for attribute $A$, $S_v$ is the subset of $S$ for which attribute $A$ has value $v$.

    *Example:* In a dataset of loan applications, a node has 50 approved and 50 rejected loans. Its entropy is high. If splitting by 'credit score > 700' results in one child node with 45 approved and 5 rejected, and another with 5 approved and 45 rejected, this split provides high information gain, as it significantly reduces uncertainty about loan approval in the child nodes.


For regression tasks, common splitting criteria include:

- **Mean Squared Error (MSE):** For a given node, MSE is the average of the squared differences between the actual and predicted values. The goal is to find a split that minimizes the weighted average MSE of the resulting child nodes. A lower MSE indicates better homogeneity within the node. $MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\bar{y})^2$ where $N$ is the number of samples, $y_i$ is the actual value, and $\bar{y}$ is the mean of actual values in the node.

### Stopping Criteria

Tree growth stops when certain conditions are met to prevent overfitting and ensure the tree remains interpretable:

* **Maximum Depth:** The maximum number of levels from the root to the furthest leaf.
* **Minimum Samples Split:** The minimum number of samples required to split an internal node.
* **Minimum Samples Leaf:** The minimum number of samples required to be at a leaf node.
* **Minimum Impurity Decrease:** A node will be split only if this split induces a decrease of the impurity greater than or equal to this value.

### Advantages and Disadvantages of Decision Trees

- **Advantages:** Simple to understand and interpret (visualizable), capable of handling both numerical and categorical data, requires little data preparation (no normalization/scaling needed), can handle multi-output problems.
- **Disadvantages:** Prone to overfitting (especially deep trees), sensitive to small variations in data (high variance), can create biased trees if some classes dominate.


## Ensemble Methods: Random Forests

Ensemble methods combine the predictions of multiple individual models to improve overall predictive performance and robustness. Random Forests are a powerful ensemble learning method specifically designed to overcome the limitations of individual decision trees, primarily their tendency to overfit and their high variance.

### The Power of Ensembles: Bagging

Random Forests use a technique called **Bagging** (Bootstrap Aggregating). It's a fancy way of saying "Democracy with a twist".

1.  **Bootstrapping (The Twist):** Each tree in the forest studies a *slightly different* version of the textbook. We create random subsets of the data (with replacement), so Tree A might see mostly customer churners, while Tree B sees mostly loyal customers.
2.  **Aggregating (The Democracy):** Once all trees have learned, they vote.
    - **Classification:** Majority wins (Vote).
    - **Regression:** Average of all predictions.

### Random Forests: Beyond Bagging

Random Forests add an additional layer of randomness to bagging:

* **Feature Randomness:** At each node split, instead of considering all features, only a random subset of features is considered for finding the best split. This further decorrelates the individual trees, making the ensemble more robust. If there is one very strong predictor, regular bagging might lead to many similar trees. Random feature selection helps to reduce this problem.

The combination of bootstrapping and feature randomness ensures that each tree in the forest is diverse and contributes uniquely to the overall prediction.

### Advantages of Random Forests

* **Reduced Overfitting:** By averaging multiple deep trees, Random Forests significantly reduce the risk of overfitting compared to a single decision tree. The ensemble smooths out the individual trees' tendency to fit noise in the training data.
* **Improved Accuracy:** The combination of diverse models generally leads to higher predictive accuracy than any single model.
* **Handles High Dimensionality:** Can effectively handle datasets with a large number of features.
* **Implicit Feature Importance:** Random Forests can provide estimates of feature importance, indicating which features contribute most to the predictions.
* **Robustness to Noise:** Less sensitive to noise in the data due to the averaging effect.

### Disadvantages of Random Forests

* **Less Interpretable:** While individual trees are interpretable, understanding the decision-making process of an entire forest with hundreds of trees is challenging.
* **Computationally More Intensive:** Training many trees requires more computational resources and time than training a single tree.
* **Can be Slower for Predictions:** Making predictions involves running data through multiple trees.

## Practical Implementation with Scikit-learn

Scikit-learn provides robust implementations for both Decision Trees and Random Forests.

### Decision Tree Implementation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Assume 'churn_data.csv' is our preprocessed Customer Churn dataset
# This dataset should be prepared following the steps from Module 2.
# It should include numerical features and a binary 'Churn' target variable (0 or 1).
df = pd.read_csv('preprocessed_churn_data.csv')

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
# max_depth controls the maximum depth of the tree to prevent overfitting.
# random_state ensures reproducibility.
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the model on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the model
print("Decision Tree Classifier Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Visualize a smaller decision tree (if max_depth is small enough)
# Note: For very deep trees, visualization can be complex.
plt.figure(figsize=(20,10))
tree.plot_tree(dt_classifier, 
               feature_names=X.columns, 
               class_names=['No Churn', 'Churn'], 
               filled=True, 
               rounded=True, 
               fontsize=10)
plt.title("Decision Tree Visualization (Max Depth = 5)")
plt.show()

# Get feature importances
print("\nDecision Tree Feature Importances:")
feature_importances_dt = pd.Series(dt_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances_dt)

```

### Random Forest Implementation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assume 'churn_data.csv' is our preprocessed Customer Churn dataset
df = pd.read_csv('preprocessed_churn_data.csv')

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
# n_estimators: The number of trees in the forest. More trees generally lead to better performance but longer training times.
# max_depth: Maximum depth of each individual tree. Limiting this can help prevent individual trees from overfitting too much.
# random_state: For reproducibility.
# class_weight: Can be used to handle imbalanced datasets by giving more weight to minority classes.
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the model
print("Random Forest Classifier Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Get feature importances from the Random Forest
print("\nRandom Forest Feature Importances:")
feature_importances_rf = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances_rf)
```

## Exercises and Practice Activities
Solutions can be found in: [`_4_3_exercise_solution.py`](_4_3_exercise_solution.py)

1. Decision Tree Hyperparameter Tuning:

    - Using the `customer_churn_preprocessed.csv` dataset, experiment with different `max_depth` values (e.g., 3, 7, 10, None) for the `DecisionTreeClassifier`.
    - Observe how the accuracy and the classification report change. What happens to the model's performance as `max_depth` increases or decreases significantly? Discuss the trade-offs between underfitting and overfitting with respect to tree depth.
    - Also, try adjusting `min_samples_leaf` (e.g., 1, 5, 10, 20) and `min_samples_split` (e.g., 2, 5, 10). How do these parameters affect the tree structure and overall performance?

2. Random Forest Hyperparameter Tuning and Comparison:

    - For the `RandomForestClassifier`, vary the `n_estimators` (e.g., 50, 200, 500) and `max_depth` (e.g., 5, 10, 20, None).
    - Compare the performance (accuracy, precision, recall, F1-score) of the best Decision Tree model you found in Exercise 1 with your tuned Random Forest model.
    - What differences do you observe in the feature importances between the single Decision Tree and the Random Forest? Explain why these differences might occur.

3. Handling Imbalanced Data:

    - Many real-world datasets, including customer churn, can be imbalanced (e.g., far fewer churners than non-churners). Check the class distribution of the 'Churn' column in your dataset.
    - If the dataset is imbalanced, re-train both the Decision Tree and Random Forest models using the `class_weight='balanced'` parameter.
    - Compare the precision, recall, and f1-score for the 'Churn' class (minority class) before and after applying `class_weight='balanced'`. Discuss the impact of this parameter on model performance for the minority class.

## Real-World Application

Decision Trees and Random Forests are widely used across various industries due to their versatility and interpretability (for single trees).



- **Customer Churn Prediction (Our Case Study):** Companies use these models to identify customers at risk of churning. For example, a telecommunications company might build a decision tree to predict which customers are likely to cancel their service based on usage patterns, contract type, and customer service interactions. The tree could show rules like "If contract type is month-to-month AND data usage is low AND has called customer service multiple times, then CHURN." Random Forests enhance this by providing more robust predictions, helping marketing teams target retention efforts effectively.

- **Medical Diagnosis:** In healthcare, decision trees can assist in diagnosing diseases. A tree might guide a doctor through a series of symptoms and test results to arrive at a probable diagnosis. For instance, a tree might start with 'fever', then branch to 'cough', then 'shortness of breath', leading to a diagnosis of 'pneumonia'. Random Forests can then aggregate many such diagnostic trees, making the overall diagnostic prediction more reliable and less prone to misdiagnosis based on subtle variations in patient data.

- **Fraud Detection:** Financial institutions use Random Forests to detect fraudulent transactions. The model analyzes various transaction attributes (amount, location, frequency, type of merchant) to identify patterns indicative of fraud. Each tree in the forest might identify a specific fraud pattern, and the collective vote of many trees provides a strong indicator of whether a transaction is legitimate or fraudulent.

- **Credit Risk Assessment:** Banks employ these algorithms to assess the creditworthiness of loan applicants. Factors like income, debt-to-income ratio, employment history, and past credit behavior are fed into the model. A decision tree could segment applicants into low, medium, or high-risk categories based on these factors, while a Random Forest provides a more nuanced and accurate risk score.

- **E-commerce Recommendation Systems:** Although more complex models like collaborative filtering are common, simpler decision-tree-like structures can be used for initial recommendations. For example, if a user buys 'Item A' and 'Item B', a tree might suggest 'Item C'. Random Forests can combine many such simple preference trees to create more sophisticated and accurate recommendations by learning from diverse user behaviors.

# Next Steps and Future Learning Directions

This lesson provided a deep dive into Decision Trees and Random Forests, including their theoretical underpinnings and practical implementation. You learned how individual trees partition data and how ensemble methods like Random Forests aggregate multiple trees to improve performance and robustness. The upcoming lessons in this module will focus on evaluating the performance of these and other machine learning models using specific metrics for regression (MAE, MSE, R2) and classification (Accuracy, Precision, Recall, F1-Score). You will then apply these classic ML algorithms to the Customer Churn Prediction Case Study, building on the implementations covered in this lesson and the data preprocessing from Module 2. This will involve selecting the most appropriate evaluation metrics based on the problem type and business goals.