"""
EXERCISE 3

Customer Churn Prediction Case Study - Initial Exploration: While churn is a classification problem, we can use linear regression to predict a propensity score for churn (a continuous 
value representing the likelihood, which we would later threshold for classification). For this exercise, load the prepared churn data from "Preparing the Customer Churn Case Study Data 
for Modeling" and perform the following:

- Select a few numerical features (e.g., MonthlyCharges, TotalCharges, tenure) and Churn_Numeric (assuming 0 for no churn, 1 for churn) as the target.
- Train a linear regression model to predict Churn_Numeric based on these features.
- Evaluate the model using MSE and R-squared.
- Discuss briefly (in comments or notes) why directly interpreting the output of a linear regression as a probability (0 to 1) for a binary outcome can be problematic (e.g., predictions 
outside 0-1 range). This will foreshadow the need for logistic regression.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("*"*80)
print("CUSTOMER CHURN PREDICTION - LINEAR REGRESSION ANALYSIS")
print("*"*80)

"""
STEP 1: Load and prepare the data
"""

print("\n" + "="*80)
print("STEP 1: DATA LOADING AND PREPARATION")
print("="*80)

# Load the CSV data
data = pd.read_csv('../../_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling/Telco-Customer-Churn.csv')

print("\nData loaded successfully!")
print(f"Dataset shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())

# Check data types and missing values
print(f"\nData types:")
print(data.dtypes)
print(f"\nMissing values:")
print(data.isnull().sum())

"""
STEP 2: Data Cleaning
"""

print("\n" + "="*80)
print("STEP 2: DATA CLEANING")
print("="*80)

# Convert TotalCharges to numeric (some may be empty strings)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Check for missing values after conversion
missing_total_charges = data['TotalCharges'].isnull().sum()
print(f"\nMissing values in TotalCharges: {missing_total_charges}")

if missing_total_charges > 0:
    # Impute missing TotalCharges with median
    median_charges = data['TotalCharges'].median()
    data['TotalCharges'].fillna(median_charges, inplace=True)
    print(f"✓ Imputed {missing_total_charges} missing values with median: {median_charges:.2f}")

# Create Churn_Numeric (0 for No, 1 for Yes)
data['Churn_Numeric'] = data['Churn'].map({'No': 0, 'Yes': 1})

print(f"\nCreated Churn_Numeric column")
print(f"\nChurn distribution:")
print(data['Churn'].value_counts())
print(f"\nChurn rate: {data['Churn_Numeric'].mean():.2%}")

"""
STEP 3: Feature Selection and Exploration
"""

print("\n" + "="*80)
print("STEP 3: FEATURE SELECTION AND EXPLORATION")
print("="*80)

# Select numerical features
numerical_features = ['MonthlyCharges', 'TotalCharges', 'tenure']
target = 'Churn_Numeric'

print(f"\nSelected features: {numerical_features}")
print(f"Target variable: {target}")

# Statistical summary of selected features
print(f"\nDescriptive statistics:")
print(data[numerical_features + [target]].describe())

# Check correlations
print(f"\nCorrelation with Churn:")
correlations = data[numerical_features].corrwith(data[target])
print(correlations.sort_values(ascending=False))

# Visualize feature distributions and relationships
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Feature Distributions and Relationships with Churn', 
             fontsize=16, fontweight='bold')

# Plot distributions for each feature by churn status
for idx, feature in enumerate(numerical_features):
    ax = axes[0, idx]
    
    # Box plot
    data.boxplot(column=feature, by='Churn', ax=ax)
    ax.set_title(f'{feature} by Churn Status')
    ax.set_xlabel('Churn')
    ax.set_ylabel(feature)
    plt.sca(ax)
    plt.xticks([1, 2], ['No', 'Yes'])

# Plot scatter plots against churn
for idx, feature in enumerate(numerical_features):
    ax = axes[1, idx]
    
    # Scatter plot with jitter for binary outcome
    churn_jitter = data[target] + np.random.normal(0, 0.02, len(data))
    ax.scatter(data[feature], churn_jitter, alpha=0.3, s=20)
    ax.set_xlabel(feature, fontweight='bold')
    ax.set_ylabel('Churn (0=No, 1=Yes)', fontweight='bold')
    ax.set_title(f'{feature} vs Churn')
    ax.set_ylim(-0.2, 1.2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nVisualizations created!")

"""
STEP 4: Prepare data for modeling
"""

print("\n" + "="*80)
print("STEP 4: PREPARE DATA FOR MODELING")
print("="*80)

# Prepare features and target
X = data[numerical_features].copy()
y = data[target].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"\nChurn rate in training set: {y_train.mean():.2%}")
print(f"Churn rate in test set: {y_test.mean():.2%}")

# Feature scaling (important for interpretation)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for better readability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_features, index=X_test.index)

print("\nFeatures scaled using StandardScaler")

"""
STEP 5: Train Linear Regression Model
"""

print("\n" + "="*80)
print("STEP 5: TRAIN LINEAR REGRESSION MODEL")
print("="*80)

# Train the model with scaled features
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\nLinear Regression model trained successfully!")

# Display model parameters
print(f"\nModel Parameters:")
print(f"   Intercept: {model.intercept_:.6f}")
print(f"\n   Coefficients:")
for feature, coef in zip(numerical_features, model.coef_):
    direction = "increases" if coef > 0 else "decreases"
    print(f"      {feature:<20}: {coef:>10.6f} (churn {direction} with this feature)")

"""
STEP 6: Make Predictions and Evaluate
"""
print("\n" + "="*80)
print("STEP 6: PREDICTIONS AND EVALUATION")
print("="*80)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("\nModel Performance Metrics:")
print(f"\n{'Metric':<25} {'Training Set':<20} {'Test Set':<20}")
print("-" * 65)
print(f"{'MSE':<25} {mse_train:<20.6f} {mse_test:<20.6f}")
print(f"{'RMSE':<25} {rmse_train:<20.6f} {rmse_test:<20.6f}")
print(f"{'MAE':<25} {mae_train:<20.6f} {mae_test:<20.6f}")
print(f"{'R² Score':<25} {r2_train:<20.6f} {r2_test:<20.6f}")

print(f"\nInterpretation:")
print(f"   • MSE = {mse_test:.6f}: Average squared error in churn prediction")
print(f"   • RMSE = {rmse_test:.6f}: Average prediction error is ±{rmse_test:.2%}")
print(f"   • R² = {r2_test:.6f}: Model explains {r2_test*100:.2f}% of variance in churn")

"""
STEP 7: Analyze Prediction Range (THE PROBLEM!)
"""

print("\n" + "="*80)
print("STEP 7: ANALYZING PREDICTIONS - THE CRITICAL PROBLEM")
print("="*80)

# Check prediction range
print(f"\nPrediction Range Analysis:")
print(f"   Training set predictions:")
print(f"      Min: {y_train_pred.min():.4f}")
print(f"      Max: {y_train_pred.max():.4f}")
print(f"      Mean: {y_train_pred.mean():.4f}")
print(f"\n   Test set predictions:")
print(f"      Min: {y_test_pred.min():.4f}")
print(f"      Max: {y_test_pred.max():.4f}")
print(f"      Mean: {y_test_pred.mean():.4f}")

# Count predictions outside [0, 1] range
below_zero = (y_test_pred < 0).sum()
above_one = (y_test_pred > 1).sum()
in_range = ((y_test_pred >= 0) & (y_test_pred <= 1)).sum()

print(f"\nPROBLEM: Predictions Outside Valid Range [0, 1]:")
print(f"   Predictions < 0: {below_zero} ({below_zero/len(y_test_pred)*100:.2f}%)")
print(f"   Predictions > 1: {above_one} ({above_one/len(y_test_pred)*100:.2f}%)")
print(f"   Predictions in [0, 1]: {in_range} ({in_range/len(y_test_pred)*100:.2f}%)")

# Show examples of problematic predictions
print(f"\nExamples of Problematic Predictions:")
problematic = pd.DataFrame({
    'MonthlyCharges': X_test.iloc[:10]['MonthlyCharges'].values,
    'TotalCharges': X_test.iloc[:10]['TotalCharges'].values,
    'tenure': X_test.iloc[:10]['tenure'].values,
    'Actual_Churn': y_test.iloc[:10].values,
    'Predicted_Score': y_test_pred[:10],
    'Valid_Range?': ['Yes' if 0 <= p <= 1 else 'NO' for p in y_test_pred[:10]]
})
print(problematic.to_string(index=False))

"""
STEP 8: Comprehensive Visualizations
"""

print("\n" + "="*80)
print("STEP 8: VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Linear Regression for Binary Classification - Problems Visualized', 
             fontsize=16, fontweight='bold')

# Plot 1: Predicted vs Actual (with problematic regions highlighted)
ax1 = axes[0, 0]
scatter = ax1.scatter(y_test, y_test_pred, c=y_test_pred, cmap='RdYlGn_r',
                     alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
ax1.plot([0, 1], [0, 1], 'b--', linewidth=2, label='Perfect prediction', alpha=0.8)
ax1.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.3)
ax1.axhline(y=1, color='red', linestyle='-', linewidth=2, alpha=0.3)
ax1.fill_between([0, 1], -0.5, 0, alpha=0.2, color='red', label='Invalid (<0)')
ax1.fill_between([0, 1], 1, 1.5, alpha=0.2, color='red', label='Invalid (>1)')
ax1.set_xlabel('Actual Churn (0=No, 1=Yes)', fontweight='bold')
ax1.set_ylabel('Predicted Churn Score', fontweight='bold')
ax1.set_title('Actual vs Predicted - Notice Invalid Range!', fontweight='bold')
ax1.set_ylim(-0.3, 1.3)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Predicted Score')

# Plot 2: Distribution of predictions
ax2 = axes[0, 1]
ax2.hist(y_test_pred, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Valid range boundary')
ax2.axvline(x=1, color='red', linestyle='--', linewidth=2)
ax2.axvspan(-1, 0, alpha=0.2, color='red', label='Invalid (<0)')
ax2.axvspan(1, 2, alpha=0.2, color='red', label='Invalid (>1)')
ax2.set_xlabel('Predicted Churn Score', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('Distribution of Predictions', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Predictions by actual class
ax3 = axes[1, 0]
no_churn_preds = y_test_pred[y_test == 0]
yes_churn_preds = y_test_pred[y_test == 1]

positions = [1, 2]
bp = ax3.boxplot([no_churn_preds, yes_churn_preds], positions=positions,
                  widths=0.6, patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)
    
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_xticks(positions)
ax3.set_xticklabels(['No Churn (0)', 'Yes Churn (1)'])
ax3.set_ylabel('Predicted Score', fontweight='bold')
ax3.set_title('Prediction Distribution by Actual Class', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.text(0.05, 0.95, f'Overlap shows\nmodel struggles', 
        transform=ax3.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 4: Feature importance (coefficients)
ax4 = axes[1, 1]
coef_df = pd.DataFrame({
    'Feature': numerical_features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
bars = ax4.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, 
                alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Coefficient Value', fontweight='bold')
ax4.set_title('Feature Coefficients (Scaled Features)', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (feature, coef) in enumerate(zip(coef_df['Feature'], coef_df['Coefficient'])):
    ax4.text(coef, i, f' {coef:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("\nVisualizations created!")

"""
STEP 9: Why Linear Regression is Problematic for Binary Classification
"""

print("\n" + "="*80)
print("STEP 9: WHY LINEAR REGRESSION IS PROBLEMATIC FOR BINARY OUTCOMES")
print("="*80)

print("\nLinear Regression is not suitable for binary classification because it can predict values outside the [0, 1] range, which is not valid for probabilities.")