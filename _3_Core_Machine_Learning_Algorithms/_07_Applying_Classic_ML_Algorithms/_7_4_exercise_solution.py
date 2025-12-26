"""
Exercise Solutions: Hyperparameter Tuning for Churn Prediction Models

This script demonstrates how to experiment with different hyperparameters for Decision Tree
and Random Forest models to improve their performance on the customer churn prediction task.

Exercises:
1. Decision Tree Hyperparameter Tuning - Experimenting with max_depth
2. Random Forest Hyperparameter Tuning - Experimenting with n_estimators and max_features
3. Feature Importance Analysis - Identifying key drivers of customer churn
4. Threshold Adjustment - Balancing precision and recall for business objectives
5. Model Comparison - Comparing all models and making business recommendations

Learning Objectives:
- Understand how hyperparameters affect model performance
- Learn to compare models using multiple evaluation metrics
- Develop intuition for choosing appropriate hyperparameter values
- Extract and interpret feature importance for business insights
- Apply threshold adjustment to optimize for business goals
- Make data-driven model selection decisions based on business objectives
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading and Preprocessing ---
# (Using the same preprocessing pipeline from the main examples)

print("="*80)
print("LOADING AND PREPROCESSING DATA")
print("="*80)

# Load the dataset
try:
    df = pd.read_csv('../../_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling/Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Dataset not found. Please ensure 'Telco-Customer-Churn.csv' is in the correct path.")
    try:
        df = pd.read_csv('Telco-Customer-Churn.csv')
    except FileNotFoundError:
        print("Could not find the dataset. Exiting...")
        exit()

# Initial Data Cleaning and Feature Engineering
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    df[col] = df[col].replace({'No phone service': 'No', 'No internet service': 'No'})

df['NumServices'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Preprocessing Setup
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
nominal_categorical_cols = [col for col in X_train.select_dtypes(include=['object']).columns.tolist() if col != 'Contract']
contract_order = [['Month-to-month', 'One year', 'Two year']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat_nominal', OneHotEncoder(handle_unknown='ignore', drop='first'), nominal_categorical_cols),
        ('cat_ordinal', OrdinalEncoder(categories=contract_order), ['Contract'])
    ],
    remainder='passthrough'
)

# Apply Preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if hasattr(X_train_processed, 'toarray'):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_test_processed, 'toarray'):
    X_test_processed = X_test_processed.toarray()

print("Data loaded and preprocessed successfully!")
print(f"Training samples: {X_train_processed.shape[0]}")
print(f"Testing samples: {X_test_processed.shape[0]}")
print(f"Number of features: {X_train_processed.shape[1]}")

# ============================================================================
# EXERCISE 1: DECISION TREE HYPERPARAMETER TUNING (max_depth)
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 1: DECISION TREE HYPERPARAMETER TUNING")
print("="*80)
print("\nExperimenting with different max_depth values...")
print("\nWhat is max_depth?")
print("- Controls how deep the tree can grow")
print("- Smaller values = simpler tree (may underfit)")
print("- Larger values = complex tree (may overfit)")
print("- None = tree grows until all leaves are pure (often overfits)")

# Test different max_depth values
max_depth_values = [3, 7, 10]
dt_results = []

for depth in max_depth_values:
    # Train Decision Tree with specific max_depth
    dt_model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=50,
        random_state=42,
        criterion='gini'
    )
    dt_model.fit(X_train_processed, y_train)
    
    # Make predictions
    y_pred = dt_model.predict(X_test_processed)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    dt_results.append({
        'max_depth': depth,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
    
    print(f"\nmax_depth = {depth}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# Find the best max_depth based on F1-score
best_dt = max(dt_results, key=lambda x: x['f1_score'])
print(f"\n{'='*80}")
print(f"Best max_depth: {best_dt['max_depth']} (F1-Score: {best_dt['f1_score']:.4f})")
print(f"{'='*80}")

print("\nObservations:")
print("- Too small max_depth (e.g., 3): Model is too simple, may miss important patterns (underfitting)")
print("- Too large max_depth (e.g., 10+): Model captures noise in training data (overfitting)")
print("- Optimal max_depth: Balances complexity and generalization")

# Visualize Decision Tree results
dt_df = pd.DataFrame(dt_results)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: All metrics vs max_depth
axes[0].plot(dt_df['max_depth'], dt_df['accuracy'], marker='o', label='Accuracy')
axes[0].plot(dt_df['max_depth'], dt_df['precision'], marker='s', label='Precision')
axes[0].plot(dt_df['max_depth'], dt_df['recall'], marker='^', label='Recall')
axes[0].plot(dt_df['max_depth'], dt_df['f1_score'], marker='d', label='F1-Score')
axes[0].set_xlabel('max_depth')
axes[0].set_ylabel('Score')
axes[0].set_title('Decision Tree: Impact of max_depth on Performance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Bar chart for best model
best_metrics = [best_dt['accuracy'], best_dt['precision'], best_dt['recall'], best_dt['f1_score']]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
axes[1].bar(metric_names, best_metrics, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
axes[1].set_ylabel('Score')
axes[1].set_title(f'Best Decision Tree (max_depth={best_dt["max_depth"]})')
axes[1].set_ylim([0, 1])
for i, v in enumerate(best_metrics):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ============================================================================
# EXERCISE 2: RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 2: RANDOM FOREST HYPERPARAMETER TUNING")
print("="*80)

# Part 2a: Tuning n_estimators
print("\n--- Part 2a: Experimenting with n_estimators ---")
print("\nWhat is n_estimators?")
print("- Number of decision trees in the forest")
print("- More trees = more stable predictions (but slower training)")
print("- Typical values: 50-500")

n_estimators_values = [50, 200, 500]
rf_n_estimators_results = []

for n_est in n_estimators_values:
    # Train Random Forest with specific n_estimators
    rf_model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train_processed, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_processed)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    rf_n_estimators_results.append({
        'n_estimators': n_est,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
    
    print(f"\nn_estimators = {n_est}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

best_n_est = max(rf_n_estimators_results, key=lambda x: x['f1_score'])
print(f"\nBest n_estimators: {best_n_est['n_estimators']} (F1-Score: {best_n_est['f1_score']:.4f})")

# Part 2b: Tuning max_features
print("\n--- Part 2b: Experimenting with max_features ---")
print("\nWhat is max_features?")
print("- Number of features to consider when looking for the best split")
print("- 'sqrt': square root of total features (good default)")
print("- 'log2': log2 of total features")
print("- 0.5: 50% of features")
print("- Lower values = more randomness, less correlation between trees")

max_features_values = ['sqrt', 'log2', 0.5]
rf_max_features_results = []

for max_feat in max_features_values:
    # Train Random Forest with specific max_features
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features=max_feat,
        random_state=42
    )
    rf_model.fit(X_train_processed, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_processed)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    rf_max_features_results.append({
        'max_features': str(max_feat),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
    
    print(f"\nmax_features = {max_feat}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

best_max_feat = max(rf_max_features_results, key=lambda x: x['f1_score'])
print(f"\nBest max_features: {best_max_feat['max_features']} (F1-Score: {best_max_feat['f1_score']:.4f})")

# Visualize Random Forest results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: n_estimators impact
rf_n_est_df = pd.DataFrame(rf_n_estimators_results)
x_pos = np.arange(len(rf_n_est_df))
width = 0.2
axes[0].bar(x_pos - 1.5*width, rf_n_est_df['accuracy'], width, label='Accuracy', color='skyblue')
axes[0].bar(x_pos - 0.5*width, rf_n_est_df['precision'], width, label='Precision', color='lightcoral')
axes[0].bar(x_pos + 0.5*width, rf_n_est_df['recall'], width, label='Recall', color='lightgreen')
axes[0].bar(x_pos + 1.5*width, rf_n_est_df['f1_score'], width, label='F1-Score', color='gold')
axes[0].set_xlabel('n_estimators')
axes[0].set_ylabel('Score')
axes[0].set_title('Random Forest: Impact of n_estimators')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(rf_n_est_df['n_estimators'])
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: max_features impact
rf_max_feat_df = pd.DataFrame(rf_max_features_results)
x_pos = np.arange(len(rf_max_feat_df))
axes[1].bar(x_pos - 1.5*width, rf_max_feat_df['accuracy'], width, label='Accuracy', color='skyblue')
axes[1].bar(x_pos - 0.5*width, rf_max_feat_df['precision'], width, label='Precision', color='lightcoral')
axes[1].bar(x_pos + 0.5*width, rf_max_feat_df['recall'], width, label='Recall', color='lightgreen')
axes[1].bar(x_pos + 1.5*width, rf_max_feat_df['f1_score'], width, label='F1-Score', color='gold')
axes[1].set_xlabel('max_features')
axes[1].set_ylabel('Score')
axes[1].set_title('Random Forest: Impact of max_features')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(rf_max_feat_df['max_features'])
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# FINAL COMPARISON: DECISION TREE vs RANDOM FOREST
# ============================================================================

print("\n" + "="*80)
print("FINAL COMPARISON: BEST DECISION TREE vs BEST RANDOM FOREST")
print("="*80)

print(f"\nBest Decision Tree (max_depth={best_dt['max_depth']}):")
print(f"  Accuracy:  {best_dt['accuracy']:.4f}")
print(f"  Precision: {best_dt['precision']:.4f}")
print(f"  Recall:    {best_dt['recall']:.4f}")
print(f"  F1-Score:  {best_dt['f1_score']:.4f}")

print(f"\nBest Random Forest (n_estimators={best_n_est['n_estimators']}):")
print(f"  Accuracy:  {best_n_est['accuracy']:.4f}")
print(f"  Precision: {best_n_est['precision']:.4f}")
print(f"  Recall:    {best_n_est['recall']:.4f}")
print(f"  F1-Score:  {best_n_est['f1_score']:.4f}")

# Comparison visualization
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
dt_scores = [best_dt['accuracy'], best_dt['precision'], best_dt['recall'], best_dt['f1_score']]
rf_scores = [best_n_est['accuracy'], best_n_est['precision'], best_n_est['recall'], best_n_est['f1_score']]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, dt_scores, width, label='Decision Tree', color='steelblue')
bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='forestgreen')

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Best Decision Tree vs Best Random Forest')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("\n1. Hyperparameter tuning significantly impacts model performance")
print("2. Decision Trees:")
print("   - max_depth controls model complexity")
print("   - Too shallow = underfitting, too deep = overfitting")
print("\n3. Random Forests:")
print("   - Generally more robust than single Decision Trees")
print("   - n_estimators: more trees = better performance (diminishing returns)")
print("   - max_features: controls randomness and tree correlation")
print("\n4. For churn prediction:")
print("   - Recall is often more important (catching actual churners)")
print("   - F1-Score balances precision and recall")
print("   - Random Forests typically outperform single Decision Trees")

# ============================================================================
# EXERCISE 3: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 3: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

print("\nWhat is Feature Importance?")
print("- Shows which features contribute most to the model's predictions")
print("- Higher importance = feature has more influence on churn prediction")
print("- Helps identify key drivers of customer churn")
print("- Can guide business decisions and intervention strategies")

# Train a Random Forest model with optimal hyperparameters
print("\nTraining Random Forest with optimal hyperparameters...")
rf_final = RandomForestClassifier(
    n_estimators=best_n_est['n_estimators'],
    max_depth=10,
    random_state=42
)
rf_final.fit(X_train_processed, y_train)

# Get feature names from the preprocessor
feature_names = preprocessor.get_feature_names_out()

# Extract feature importances
feature_importances = pd.Series(
    rf_final.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\n" + "-"*80)
print("TOP 10 MOST IMPORTANT FEATURES")
print("-"*80)

for i, (feature, importance) in enumerate(feature_importances.head(10).items(), 1):
    print(f"{i:2d}. {feature:45s} {importance:.6f} ({importance*100:.2f}%)")

print("\n" + "-"*80)
print("TOP 5 FEATURES (DETAILED ANALYSIS)")
print("-"*80)

top_5_features = feature_importances.head(5)

# Provide business interpretation for top features
feature_interpretations = {
    'tenure': "Customer tenure (how long they've been with the company)",
    'Contract': "Type of contract (month-to-month, one year, two year)",
    'TotalCharges': "Total amount charged to the customer over their lifetime",
    'MonthlyCharges': "Monthly subscription fee",
    'InternetService': "Type of internet service (DSL, Fiber optic, None)",
    'OnlineSecurity': "Whether customer has online security service",
    'OnlineBackup': "Whether customer has online backup service",
    'TechSupport': "Whether customer has tech support service",
    'PaymentMethod': "How customer pays their bill",
    'PaperlessBilling': "Whether customer uses paperless billing",
    'NumServices': "Total number of services subscribed to"
}

for i, (feature, importance) in enumerate(top_5_features.items(), 1):
    # Extract the base feature name (remove prefix like 'num__', 'cat_nominal__', etc.)
    base_feature = feature.split('__')[-1]
    
    # Find matching interpretation
    interpretation = None
    for key, value in feature_interpretations.items():
        if key.lower() in base_feature.lower():
            interpretation = value
            break
    
    print(f"\n{i}. {feature}")
    print(f"   Importance: {importance:.6f} ({importance*100:.2f}%)")
    if interpretation:
        print(f"   Meaning: {interpretation}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Top 10 Feature Importances (Horizontal Bar Chart)
ax1 = axes[0, 0]
top_10 = feature_importances.head(10)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10)))
ax1.barh(range(len(top_10)), top_10.values, color=colors)
ax1.set_yticks(range(len(top_10)))
ax1.set_yticklabels(top_10.index, fontsize=9)
ax1.set_xlabel('Importance Score')
ax1.set_title('Top 10 Most Important Features for Churn Prediction')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(top_10.values):
    ax1.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=8)

# Plot 2: Top 5 Features (Pie Chart)
ax2 = axes[0, 1]
top_5 = feature_importances.head(5)
others = feature_importances[5:].sum()
pie_data = list(top_5.values) + [others]
pie_labels = [f.split('__')[-1][:20] for f in top_5.index] + ['Others']
colors_pie = plt.cm.Set3(range(len(pie_data)))

wedges, texts, autotexts = ax2.pie(
    pie_data,
    labels=pie_labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_pie
)
ax2.set_title('Feature Importance Distribution\n(Top 5 vs Others)')

# Make percentage text more readable
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

# Plot 3: Cumulative Importance
ax3 = axes[1, 0]
cumulative_importance = feature_importances.cumsum()
ax3.plot(range(len(cumulative_importance)), cumulative_importance.values, 
         marker='o', markersize=3, linewidth=2, color='steelblue')
ax3.axhline(y=0.8, color='red', linestyle='--', label='80% threshold')
ax3.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
ax3.set_xlabel('Number of Features')
ax3.set_ylabel('Cumulative Importance')
ax3.set_title('Cumulative Feature Importance')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Find how many features needed for 80% and 90%
features_80 = (cumulative_importance >= 0.8).idxmax()
features_90 = (cumulative_importance >= 0.9).idxmax()
n_features_80 = list(cumulative_importance.index).index(features_80) + 1
n_features_90 = list(cumulative_importance.index).index(features_90) + 1

ax3.text(0.5, 0.5, f'{n_features_80} features\nfor 80% importance', 
         transform=ax3.transAxes, fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Feature Importance by Category
ax4 = axes[1, 1]

# Categorize features
numerical_importance = feature_importances[feature_importances.index.str.startswith('num__')].sum()
categorical_nominal_importance = feature_importances[feature_importances.index.str.startswith('cat_nominal__')].sum()
categorical_ordinal_importance = feature_importances[feature_importances.index.str.startswith('cat_ordinal__')].sum()

categories = ['Numerical\nFeatures', 'Categorical\n(Nominal)', 'Categorical\n(Ordinal)']
importances = [numerical_importance, categorical_nominal_importance, categorical_ordinal_importance]
colors_cat = ['skyblue', 'lightcoral', 'lightgreen']

bars = ax4.bar(categories, importances, color=colors_cat, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Total Importance')
ax4.set_title('Feature Importance by Feature Type')
ax4.set_ylim([0, max(importances) * 1.2])
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Business Insights
print("\n" + "="*80)
print("BUSINESS INSIGHTS FROM FEATURE IMPORTANCE")
print("="*80)

print("\nKey Findings:")
print(f"1. Top feature accounts for {feature_importances.iloc[0]*100:.2f}% of importance")
print(f"2. Top 5 features account for {top_5.sum()*100:.2f}% of total importance")
print(f"3. {n_features_80} features are needed to explain 80% of churn behavior")
print(f"4. {n_features_90} features are needed to explain 90% of churn behavior")

print("\nCustomer Behavior Insights:")

# Analyze top features and provide insights
top_feature = feature_importances.index[0]
if 'tenure' in top_feature.lower():
    print("\n✓ TENURE is the most important factor:")
    print("  - Newer customers are more likely to churn")
    print("  - Focus retention efforts on customers in their first year")
    print("  - Consider onboarding programs and early engagement strategies")

if any('contract' in f.lower() for f in feature_importances.head(5).index):
    print("\n✓ CONTRACT TYPE is highly influential:")
    print("  - Month-to-month contracts have higher churn risk")
    print("  - Incentivize customers to sign longer-term contracts")
    print("  - Offer discounts for annual or 2-year commitments")

if any('totalcharges' in f.lower() or 'monthlycharges' in f.lower() for f in feature_importances.head(5).index):
    print("\n✓ CHARGES (Total/Monthly) are significant:")
    print("  - Price sensitivity affects churn decisions")
    print("  - Consider tiered pricing or loyalty discounts")
    print("  - Monitor customers with high monthly charges")

if any('internet' in f.lower() for f in feature_importances.head(5).index):
    print("\n✓ INTERNET SERVICE type matters:")
    print("  - Fiber optic customers may have different churn patterns")
    print("  - Ensure service quality matches customer expectations")
    print("  - Address connectivity issues promptly")

if any('support' in f.lower() or 'security' in f.lower() or 'backup' in f.lower() for f in feature_importances.head(5).index):
    print("\n✓ ADDITIONAL SERVICES impact retention:")
    print("  - Customers with more services are less likely to churn")
    print("  - Cross-sell and upsell relevant services")
    print("  - Bundle services for better value proposition")

print("\nRecommended Actions:")
print("1. Prioritize retention efforts based on top features")
print("2. Create targeted intervention programs for high-risk segments")
print("3. Monitor these key features in real-time dashboards")
print("4. A/B test interventions focusing on top churn drivers")
print("5. Regularly update feature importance as business evolves")

print("\n" + "="*80)
print("EXERCISE 3 COMPLETE")
print("="*80)

# ============================================================================
# EXERCISE 4: THRESHOLD ADJUSTMENT FOR CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 4: THRESHOLD ADJUSTMENT")
print("="*80)

print("\nWhat is Classification Threshold?")
print("- Default threshold is 0.5: if probability >= 0.5, predict positive class (churn)")
print("- Adjusting threshold allows us to balance precision and recall")
print("- Lower threshold (e.g., 0.3): More predictions as positive → Higher Recall, Lower Precision")
print("- Higher threshold (e.g., 0.7): Fewer predictions as positive → Lower Recall, Higher Precision")
print("\nWhy adjust threshold?")
print("- In churn prediction, missing a churner (False Negative) is often costly")
print("- Business may prefer to contact more customers (lower threshold) to catch more churners")
print("- This is a business decision based on cost-benefit analysis")

# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression

print("\nTraining Logistic Regression model...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_processed, y_train)

# Get probability predictions
y_pred_proba_lr = lr_model.predict_proba(X_test_processed)[:, 1]

print("\n" + "-"*80)
print("TESTING DIFFERENT THRESHOLDS")
print("-"*80)

# Test different thresholds
thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
threshold_results = []

for threshold in thresholds_to_test:
    # Apply custom threshold
    y_pred_custom = (y_pred_proba_lr >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_custom)
    precision = precision_score(y_test, y_pred_custom, zero_division=0)
    recall = recall_score(y_test, y_pred_custom, zero_division=0)
    f1 = f1_score(y_test, y_pred_custom, zero_division=0)
    
    # Count predictions
    num_predicted_churn = y_pred_custom.sum()
    num_actual_churn = y_test.sum()
    
    threshold_results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predicted_churn': num_predicted_churn,
        'actual_churn': num_actual_churn
    })
    
    print(f"\nThreshold = {threshold:.1f}")
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall:           {recall:.4f}")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  Predicted Churn:  {num_predicted_churn}/{len(y_test)} customers")

# Create DataFrame for analysis
threshold_df = pd.DataFrame(threshold_results)

print("\n" + "-"*80)
print("THRESHOLD COMPARISON SUMMARY")
print("-"*80)
print(threshold_df.to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Precision-Recall Tradeoff
ax1 = axes[0, 0]
ax1.plot(threshold_df['threshold'], threshold_df['precision'], 
         marker='o', linewidth=2, markersize=8, label='Precision', color='coral')
ax1.plot(threshold_df['threshold'], threshold_df['recall'], 
         marker='s', linewidth=2, markersize=8, label='Recall', color='skyblue')
ax1.plot(threshold_df['threshold'], threshold_df['f1_score'], 
         marker='^', linewidth=2, markersize=8, label='F1-Score', color='lightgreen')
ax1.set_xlabel('Classification Threshold')
ax1.set_ylabel('Score')
ax1.set_title('Precision-Recall Tradeoff vs Threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')

# Plot 2: Number of Predicted Churners
ax2 = axes[0, 1]
bars = ax2.bar(threshold_df['threshold'].astype(str), threshold_df['predicted_churn'], 
               color='steelblue', edgecolor='black', linewidth=1.5)
ax2.axhline(y=threshold_df['actual_churn'].iloc[0], color='red', 
            linestyle='--', linewidth=2, label=f'Actual Churners ({threshold_df["actual_churn"].iloc[0]})')
ax2.set_xlabel('Classification Threshold')
ax2.set_ylabel('Number of Customers')
ax2.set_title('Predicted Churners at Different Thresholds')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Plot 3: ROC Curve
from sklearn.metrics import roc_curve, auc

ax3 = axes[1, 0]
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba_lr)
roc_auc = auc(fpr, tpr)

ax3.plot(fpr, tpr, color='darkorange', linewidth=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--', label='Random Classifier')

# Mark the tested thresholds on ROC curve
for threshold in thresholds_to_test:
    y_pred_temp = (y_pred_proba_lr >= threshold).astype(int)
    fpr_point = ((1 - y_test) & y_pred_temp).sum() / (1 - y_test).sum()
    tpr_point = (y_test & y_pred_temp).sum() / y_test.sum()
    ax3.plot(fpr_point, tpr_point, 'ro', markersize=8)
    ax3.annotate(f'{threshold:.1f}', (fpr_point, tpr_point), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate (Recall)')
ax3.set_title('ROC Curve with Threshold Markers')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# Plot 4: Precision-Recall Curve
from sklearn.metrics import precision_recall_curve

ax4 = axes[1, 1]
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba_lr)

ax4.plot(recall_curve, precision_curve, color='purple', linewidth=2, 
         label='Precision-Recall Curve')

# Mark the tested thresholds
for threshold in thresholds_to_test:
    result = threshold_df[threshold_df['threshold'] == threshold].iloc[0]
    ax4.plot(result['recall'], result['precision'], 'ro', markersize=8)
    ax4.annotate(f'{threshold:.1f}', (result['recall'], result['precision']), 
                xytext=(5, -10), textcoords='offset points', fontsize=9)

ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve with Threshold Markers')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Business Decision Analysis
print("\n" + "="*80)
print("BUSINESS DECISION ANALYSIS")
print("="*80)

print("\nScenario Analysis:")

# Scenario 1: Conservative (High Precision)
conservative_threshold = 0.7
conservative_result = threshold_df[threshold_df['threshold'] == conservative_threshold].iloc[0]
print(f"\n1. CONSERVATIVE APPROACH (Threshold = {conservative_threshold})")
print(f"   Goal: High confidence in predictions, minimize false alarms")
print(f"   Precision: {conservative_result['precision']:.2%} - When we predict churn, we're right {conservative_result['precision']:.0%} of the time")
print(f"   Recall: {conservative_result['recall']:.2%} - We catch {conservative_result['recall']:.0%} of actual churners")
print(f"   Impact: Contact {conservative_result['predicted_churn']} customers (may miss some churners)")
print(f"   Best for: Limited retention budget, high cost per intervention")

# Scenario 2: Balanced (Default)
balanced_threshold = 0.5
balanced_result = threshold_df[threshold_df['threshold'] == balanced_threshold].iloc[0]
print(f"\n2. BALANCED APPROACH (Threshold = {balanced_threshold})")
print(f"   Goal: Balance between precision and recall")
print(f"   Precision: {balanced_result['precision']:.2%}")
print(f"   Recall: {balanced_result['recall']:.2%}")
print(f"   F1-Score: {balanced_result['f1_score']:.4f} (harmonic mean)")
print(f"   Impact: Contact {balanced_result['predicted_churn']} customers")
print(f"   Best for: Standard operations, moderate budget")

# Scenario 3: Aggressive (High Recall)
aggressive_threshold = 0.3
aggressive_result = threshold_df[threshold_df['threshold'] == aggressive_threshold].iloc[0]
print(f"\n3. AGGRESSIVE APPROACH (Threshold = {aggressive_threshold})")
print(f"   Goal: Catch as many churners as possible")
print(f"   Precision: {aggressive_result['precision']:.2%} - More false positives (non-churners contacted)")
print(f"   Recall: {aggressive_result['recall']:.2%} - We catch {aggressive_result['recall']:.0%} of actual churners")
print(f"   Impact: Contact {aggressive_result['predicted_churn']} customers (cast wider net)")
print(f"   Best for: High customer lifetime value, low cost per intervention")

print("\n" + "-"*80)
print("COST-BENEFIT EXAMPLE")
print("-"*80)

# Example cost-benefit calculation
retention_cost_per_customer = 50  # Cost to contact and offer retention incentive
customer_lifetime_value = 1000    # Average CLV
churn_prevention_rate = 0.3       # 30% of contacted churners are retained

print(f"\nAssumptions:")
print(f"  - Cost per retention attempt: ${retention_cost_per_customer}")
print(f"  - Customer Lifetime Value: ${customer_lifetime_value}")
print(f"  - Success rate of retention: {churn_prevention_rate:.0%}")

for idx, row in threshold_df.iterrows():
    threshold = row['threshold']
    predicted_churn = row['predicted_churn']
    recall = row['recall']
    precision = row['precision']
    
    # Calculate costs and benefits
    total_cost = predicted_churn * retention_cost_per_customer
    
    # True positives (actual churners we correctly identified)
    actual_churners = threshold_df['actual_churn'].iloc[0]
    true_positives = recall * actual_churners
    
    # Customers saved
    customers_saved = true_positives * churn_prevention_rate
    
    # Revenue saved
    revenue_saved = customers_saved * customer_lifetime_value
    
    # Net benefit
    net_benefit = revenue_saved - total_cost
    roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0
    
    print(f"\nThreshold {threshold:.1f}:")
    print(f"  Customers contacted: {predicted_churn}")
    print(f"  Total cost: ${total_cost:,.0f}")
    print(f"  Churners caught: {true_positives:.1f}")
    print(f"  Customers saved: {customers_saved:.1f}")
    print(f"  Revenue saved: ${revenue_saved:,.0f}")
    print(f"  Net benefit: ${net_benefit:,.0f}")
    print(f"  ROI: {roi:.1f}%")

# Find best threshold by ROI
threshold_df['roi'] = threshold_df.apply(
    lambda row: ((row['recall'] * row['actual_churn'] * churn_prevention_rate * customer_lifetime_value) - 
                 (row['predicted_churn'] * retention_cost_per_customer)) / 
                (row['predicted_churn'] * retention_cost_per_customer) * 100 
                if row['predicted_churn'] > 0 else 0, 
    axis=1
)

best_roi_threshold = threshold_df.loc[threshold_df['roi'].idxmax()]

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"\nBased on the cost-benefit analysis:")
print(f"  Best threshold: {best_roi_threshold['threshold']:.1f}")
print(f"  Expected ROI: {best_roi_threshold['roi']:.1f}%")
print(f"  Customers to contact: {best_roi_threshold['predicted_churn']}")
print(f"  Recall: {best_roi_threshold['recall']:.2%}")
print(f"  Precision: {best_roi_threshold['precision']:.2%}")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("\n1. Classification threshold is a business decision, not just a technical one")
print("2. Lower threshold → Higher Recall (catch more churners) but Lower Precision (more false alarms)")
print("3. Higher threshold → Higher Precision (fewer false alarms) but Lower Recall (miss some churners)")
print("4. The optimal threshold depends on:")
print("   - Cost of retention efforts")
print("   - Customer lifetime value")
print("   - Success rate of retention programs")
print("   - Business priorities (growth vs. efficiency)")
print("5. Use ROC and Precision-Recall curves to visualize tradeoffs")
print("6. Always validate threshold choice with business stakeholders")


print("\n" + "="*80)
print("EXERCISE 4 COMPLETE")
print("="*80)

# ============================================================================
# EXERCISE 5: MODEL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("EXERCISE 5: COMPREHENSIVE MODEL COMPARISON")
print("="*80)

print("\nComparing all three models: Logistic Regression, Decision Tree, and Random Forest")
print("Goal: Determine which model is best for minimizing actual churn")

# Train all three models on the same data
print("\nTraining all models with consistent preprocessing...")

# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_final = LogisticRegression(random_state=42, max_iter=1000)
lr_final.fit(X_train_processed, y_train)
y_pred_lr = lr_final.predict(X_test_processed)

# 2. Decision Tree (using best hyperparameters from Exercise 1)
dt_final = DecisionTreeClassifier(
    max_depth=best_dt['max_depth'],
    min_samples_leaf=50,
    random_state=42,
    criterion='gini'
)
dt_final.fit(X_train_processed, y_train)
y_pred_dt = dt_final.predict(X_test_processed)

# 3. Random Forest (using best hyperparameters from Exercise 2)
rf_final_comparison = RandomForestClassifier(
    n_estimators=best_n_est['n_estimators'],
    max_depth=10,
    random_state=42
)
rf_final_comparison.fit(X_train_processed, y_train)
y_pred_rf = rf_final_comparison.predict(X_test_processed)

# Calculate metrics for all models
models_comparison = []

# Logistic Regression metrics
lr_metrics = {
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-Score': f1_score(y_test, y_pred_lr)
}
models_comparison.append(lr_metrics)

# Decision Tree metrics
dt_metrics = {
    'Model': 'Decision Tree',
    'Accuracy': accuracy_score(y_test, y_pred_dt),
    'Precision': precision_score(y_test, y_pred_dt),
    'Recall': recall_score(y_test, y_pred_dt),
    'F1-Score': f1_score(y_test, y_pred_dt)
}
models_comparison.append(dt_metrics)

# Random Forest metrics
rf_metrics = {
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1-Score': f1_score(y_test, y_pred_rf)
}
models_comparison.append(rf_metrics)

# Create comparison DataFrame
comparison_df = pd.DataFrame(models_comparison)

print("\n" + "-"*80)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("-"*80)
print(comparison_df.to_string(index=False))

# Identify best model for each metric
print("\n" + "-"*80)
print("BEST MODEL BY METRIC")
print("-"*80)

for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    best_model = comparison_df.loc[comparison_df[metric].idxmax()]
    print(f"{metric:12s}: {best_model['Model']:20s} ({best_model[metric]:.4f})")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Grouped Bar Chart - All Metrics
ax1 = axes[0, 0]
x = np.arange(len(comparison_df))
width = 0.2
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

for i, metric in enumerate(metrics_to_plot):
    offset = (i - 1.5) * width
    bars = ax1.bar(x + offset, comparison_df[metric], width, 
                   label=metric, color=colors[i], edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Model')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison - All Metrics')
ax1.set_xticks(x)
ax1.set_xticklabels(comparison_df['Model'])
ax1.legend()
ax1.set_ylim([0, 1.1])
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Radar Chart
ax2 = axes[0, 1]
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
num_vars = len(categories)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

ax2 = plt.subplot(2, 2, 2, projection='polar')
colors_radar = ['steelblue', 'forestgreen', 'coral']

for idx, row in comparison_df.iterrows():
    values = row[categories].tolist()
    values += values[:1]  # Complete the circle
    ax2.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors_radar[idx])
    ax2.fill(angles, values, alpha=0.15, color=colors_radar[idx])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories)
ax2.set_ylim(0, 1)
ax2.set_title('Model Performance Radar Chart', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax2.grid(True)

# Plot 3: Recall Comparison (Most Important for Churn)
ax3 = axes[1, 0]
bars = ax3.barh(comparison_df['Model'], comparison_df['Recall'], 
                color=['steelblue', 'forestgreen', 'coral'], edgecolor='black', linewidth=2)
ax3.set_xlabel('Recall Score')
ax3.set_title('Recall Comparison (Key Metric for Churn Prevention)')
ax3.set_xlim([0, 1])
ax3.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=11)

# Highlight best recall
best_recall_idx = comparison_df['Recall'].idxmax()
bars[best_recall_idx].set_color('gold')
bars[best_recall_idx].set_edgecolor('darkgoldenrod')
bars[best_recall_idx].set_linewidth(3)

# Plot 4: Confusion Matrices Comparison
ax4 = axes[1, 1]
ax4.axis('off')

# Create mini confusion matrices
conf_matrices = [
    confusion_matrix(y_test, y_pred_lr),
    confusion_matrix(y_test, y_pred_dt),
    confusion_matrix(y_test, y_pred_rf)
]

# Display confusion matrices as text
y_offset = 0.9
for idx, (model_name, cm) in enumerate(zip(comparison_df['Model'], conf_matrices)):
    text = f"{model_name}:\n"
    text += f"TN={cm[0,0]:<3} FP={cm[0,1]:<3}\n"
    text += f"FN={cm[1,0]:<3} TP={cm[1,1]:<3}\n"
    
    # Calculate False Negative Rate (missed churners)
    fn_rate = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    text += f"Missed Churners: {fn_rate:.1%}"
    
    ax4.text(0.1, y_offset - (idx * 0.3), text, 
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor=colors_radar[idx], alpha=0.3))

ax4.set_title('Confusion Matrices Comparison\n(TN=True Neg, FP=False Pos, FN=False Neg, TP=True Pos)', 
             fontsize=12, pad=10)

plt.tight_layout()
plt.show()

# Detailed Analysis
print("\n" + "="*80)
print("DETAILED ANALYSIS FOR CHURN MINIMIZATION")
print("="*80)

print("\nKey Considerations for Churn Prediction:")
print("1. RECALL is most critical - we want to catch as many actual churners as possible")
print("2. Missing a churner (False Negative) is costly - lost customer lifetime value")
print("3. Contacting a non-churner (False Positive) is less costly - just wasted effort")
print("4. F1-Score balances precision and recall")

# Calculate additional metrics
for idx, row in comparison_df.iterrows():
    model_name = row['Model']
    
    if model_name == 'Logistic Regression':
        cm = confusion_matrix(y_test, y_pred_lr)
    elif model_name == 'Decision Tree':
        cm = confusion_matrix(y_test, y_pred_dt)
    else:  # Random Forest
        cm = confusion_matrix(y_test, y_pred_rf)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n{model_name}:")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Churners Caught: {tp}/{tp+fn} ({row['Recall']:.1%})")
    print(f"  Churners Missed: {fn}/{tp+fn} ({false_negative_rate:.1%}) ← CRITICAL METRIC")
    print(f"  False Alarms: {fp}/{fp+tn} ({false_positive_rate:.1%})")

# Business Impact Analysis
print("\n" + "-"*80)
print("BUSINESS IMPACT SIMULATION")
print("-"*80)

# Assumptions
customer_lifetime_value = 1000
retention_cost = 50
retention_success_rate = 0.3
total_customers = len(y_test)
actual_churners = y_test.sum()

print(f"\nAssumptions:")
print(f"  Total customers in test set: {total_customers}")
print(f"  Actual churners: {actual_churners}")
print(f"  Customer Lifetime Value: ${customer_lifetime_value}")
print(f"  Cost per retention attempt: ${retention_cost}")
print(f"  Retention success rate: {retention_success_rate:.0%}")

business_results = []

for idx, row in comparison_df.iterrows():
    model_name = row['Model']
    
    if model_name == 'Logistic Regression':
        cm = confusion_matrix(y_test, y_pred_lr)
    elif model_name == 'Decision Tree':
        cm = confusion_matrix(y_test, y_pred_dt)
    else:
        cm = confusion_matrix(y_test, y_pred_rf)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate business metrics
    customers_contacted = tp + fp
    total_cost = customers_contacted * retention_cost
    
    # Revenue saved from successfully retained customers
    customers_saved = tp * retention_success_rate
    revenue_saved = customers_saved * customer_lifetime_value
    
    # Revenue lost from missed churners
    revenue_lost = fn * customer_lifetime_value
    
    # Net benefit
    net_benefit = revenue_saved - total_cost
    
    # ROI
    roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0
    
    business_results.append({
        'Model': model_name,
        'Contacted': customers_contacted,
        'Cost': total_cost,
        'Churners_Caught': tp,
        'Churners_Missed': fn,
        'Revenue_Saved': revenue_saved,
        'Revenue_Lost': revenue_lost,
        'Net_Benefit': net_benefit,
        'ROI': roi
    })
    
    print(f"\n{model_name}:")
    print(f"  Customers contacted: {customers_contacted}")
    print(f"  Total cost: ${total_cost:,.0f}")
    print(f"  Churners caught: {tp} (saved {customers_saved:.1f} customers)")
    print(f"  Revenue saved: ${revenue_saved:,.0f}")
    print(f"  Churners missed: {fn}")
    print(f"  Revenue lost: ${revenue_lost:,.0f}")
    print(f"  Net benefit: ${net_benefit:,.0f}")
    print(f"  ROI: {roi:.1f}%")

business_df = pd.DataFrame(business_results)

# Find best model by different criteria
best_by_revenue_saved = business_df.loc[business_df['Revenue_Saved'].idxmax()]
best_by_roi = business_df.loc[business_df['ROI'].idxmax()]
best_by_net_benefit = business_df.loc[business_df['Net_Benefit'].idxmax()]
least_revenue_lost = business_df.loc[business_df['Revenue_Lost'].idxmin()]

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print(f"\nBest model by Revenue Saved: {best_by_revenue_saved['Model']}")
print(f"Best model by ROI: {best_by_roi['Model']}")
print(f"Best model by Net Benefit: {best_by_net_benefit['Model']}")
print(f"Best model by Least Revenue Lost: {least_revenue_lost['Model']}")

# Final recommendation
best_recall_model = comparison_df.loc[comparison_df['Recall'].idxmax()]

print("\n" + "-"*80)
print("FINAL RECOMMENDATION FOR CHURN MINIMIZATION")
print("-"*80)

print(f"\n🏆 RECOMMENDED MODEL: {best_recall_model['Model']}")
print(f"\nRationale:")
print(f"1. Highest Recall: {best_recall_model['Recall']:.2%}")
print(f"   - Catches the most actual churners")
print(f"   - Minimizes revenue loss from missed churners")

print(f"\n2. Performance Metrics:")
print(f"   - Accuracy:  {best_recall_model['Accuracy']:.4f}")
print(f"   - Precision: {best_recall_model['Precision']:.4f}")
print(f"   - Recall:    {best_recall_model['Recall']:.4f}")
print(f"   - F1-Score:  {best_recall_model['F1-Score']:.4f}")

# Get business metrics for recommended model
recommended_business = business_df[business_df['Model'] == best_recall_model['Model']].iloc[0]
print(f"\n3. Business Impact:")
print(f"   - Revenue saved: ${recommended_business['Revenue_Saved']:,.0f}")
print(f"   - Revenue lost: ${recommended_business['Revenue_Lost']:,.0f}")
print(f"   - Net benefit: ${recommended_business['Net_Benefit']:,.0f}")
print(f"   - ROI: {recommended_business['ROI']:.1f}%")

print("\n4. Why this model?")
if best_recall_model['Model'] == 'Random Forest':
    print("   ✓ Ensemble method reduces overfitting")
    print("   ✓ Handles complex feature interactions well")
    print("   ✓ More robust than single Decision Tree")
    print("   ✓ Provides feature importance for business insights")
elif best_recall_model['Model'] == 'Logistic Regression':
    print("   ✓ Simple and interpretable")
    print("   ✓ Fast to train and predict")
    print("   ✓ Coefficients show feature impact")
    print("   ✓ Good baseline model")
else:  # Decision Tree
    print("   ✓ Highly interpretable decision rules")
    print("   ✓ Easy to explain to stakeholders")
    print("   ✓ Handles non-linear relationships")
    print("   ✓ Visual representation available")

print("\n5. Implementation Recommendations:")
print("   • Deploy this model for production churn prediction")
print("   • Monitor performance regularly and retrain as needed")
print("   • Consider threshold adjustment (Exercise 4) to optimize further")
print("   • Use feature importance (Exercise 3) to guide retention strategies")
print("   • A/B test retention campaigns based on model predictions")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("\n1. For churn minimization, RECALL is the most important metric")
print("2. Random Forest typically offers the best balance of performance and robustness")
print("3. Model selection should consider both technical metrics AND business impact")
print("4. The 'best' model depends on business priorities:")
print("   - High CLV customers → prioritize Recall (catch all churners)")
print("   - Limited budget → prioritize Precision (avoid false alarms)")
print("   - Balanced approach → optimize F1-Score")
print("5. Always validate model choice with business stakeholders")
print("6. Combine model predictions with domain expertise for best results")

print("\n" + "="*80)
print("EXERCISE 5 COMPLETE")
print("="*80)
print("\n" + "="*80)
print("ALL EXERCISES COMPLETE!")
print("="*80)
