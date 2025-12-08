"""
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'Gender': ['Male', 'Female', 'MALE ', 'Female', 'Male', 'Female', 'male', 'Female', 'Male', 'Female', 'Male', 'Female', 'FEMALE', 'Male', 'Female'],
    'SeniorCitizen': [0, 0, 1, 'No', 0, 1, 0, 1, 0, 0, 1, 0, 0, 'Yes', 0],
    'Partner': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Dependents': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes'],
    'Tenure': [24, 1, 10, 50, 8, 72, 5, 30, 12, 6, -2, 45, 1, 60, 3], # Added a negative tenure
    'MonthlyCharges': [70.0, 29.85, 104.80, 80.0, 99.65, 110.0, np.nan, 75.0, 90.0, 60.0, 150.0, 25.0, 120.0, 10.0, 85.0], # Added a very low charge
    'TotalCharges': [1676.0, 29.85, np.nan, 3950.0, 800.0, 7933.0, 350.0, 2250.0, np.nan, 360.0, 1800.0, 1125.0, 120.0, 600.0, 255.0],
    'Contract': ['Month-to-month', 'Month-to-month', 'Month-to-month', 'Two year', 'Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Month-to-month', 'Month-to-month', 'Two year', 'one year', 'Month-to-month', 'Two year', 'Month-to-month'],
    'Churn': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', np.nan, 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check']
}

### Exercise 1: Brainstorming Feature Engineering

Think about the Customer Churn dataset we've been using. We have the raw data, but what *stories* are hidden inside?

**Your Mission:**
Brainstorm **3-5 new features** that you could create from the existing columns. Don't worry about writing code yet—just focus on the logic.

**For each feature, ask yourself:**
1.  **Name:** What would you call it?
2.  **Recipe:** Which existing columns do you need to cook it up?
3.  **Why:** Why do you think this feature would help predict if a customer is leaving?

**Example:**
*   **New Feature:** `Has_Family`
*   **Recipe:** `Partner` + `Dependents`
*   **Why:** A customer with a full house might likely stick around longer than a single user due to higher switching effort.

---

### INSTRUCTOR SOLUTIONS (Brainstorming)

Here are a few ideas to get your creative juices flowing:

**1. Tenure Group**
*   **Derived From:** `Tenure`
*   **Rationale:** "12 months" vs "13 months" might not matter, but "New Customer" vs "Loyal Customer" definitely does. Grouping tenure (e.g., 0-12 months = "New") helps the model digest this.

**2. Service Bundle Count**
*   **Derived From:** `PhoneService`, `InternetService`, `StreamingTV`, etc.
*   **Rationale:** The simple question: "How deeply enmeshed is this customer?" Someone with 5 services is way harder to poach than someone with just one.

**3. Charge Consistency Ratio**
*   **Derived From:** `TotalCharges` / (`MonthlyCharges` * `Tenure`)
*   **Rationale:** Does the math add up? If this ratio is way off, it might mean billing errors or weird promo schemes. Confused customers = Unhappy customers.

**4. Payment Convenience Score**
*   **Derived From:** `PaymentMethod`, `PaperlessBilling`
*   **Rationale:** Is it a pain to pay the bill? Automatic credit card payments are frictionless. Mailed checks require effort. Friction leads to churn.

**5. Senior Family Support**
*   **Derived From:** `SeniorCitizen`, `Partner`, `Dependents`
*   **Rationale:** Seniors living alone might struggle with tech support and leave. Seniors living with family might get help and stay. This interaction could be key.


"""
"""
Exercise 2: Feature Selection Rationale (Churn Prediction)

Consider the original features and the newly engineered features from Exercise 1.

    Identify at least two pairs of features that might be highly correlated and explain why you might choose to keep one and remove the other in a feature selection process.
    Identify at least two features (from original or engineered) that you might consider removing due to very low variance or perceived irrelevance based on common sense (before any statistical tests). Explain your reasoning.


Part 1: Highly Correlated Feature Pairs
Based on the original features (e.g., Tenure, MonthlyCharges, TotalCharges) and the engineered features from Exercise 1 (e.g., Tenure_Group, Charge_Consistency_Ratio), 
here are two pairs that might exhibit high correlation. For each pair, I'll explain the potential correlation and rationale for keeping one while removing the other during 
feature selection (e.g., to reduce multicollinearity, which can inflate model variance and lead to overfitting).

Pair: Tenure and TotalCharges

Potential Correlation: These are likely highly correlated because TotalCharges is typically calculated as MonthlyCharges multiplied by Tenure (with minor adjustments for promotions or 
changes). In the provided dataset, this relationship holds for most entries (e.g., for CustomerID 1: 70.0 * 24 ≈ 1680, close to 1676.0).
Rationale for Selection: I would keep Tenure and remove TotalCharges. Tenure directly captures customer loyalty and recency, which are strong predictors of churn 
(e.g., new customers churn more). TotalCharges is redundant and can be derived from Tenure and MonthlyCharges, so removing it avoids multicollinearity without losing predictive power. 
If the model needs total value, it could be recalculated.
Pair: Tenure and Tenure_Group

Potential Correlation: Tenure_Group is engineered directly from Tenure by binning it into categories (e.g., "New" for 0-12 months). This creates a near-perfect correlation since 
Tenure_Group is a discretized version of Tenure.
Rationale for Selection: I would keep Tenure_Group and remove Tenure. Tenure_Group simplifies the continuous Tenure variable into interpretable categories, which can better capture 
non-linear churn patterns (e.g., "New" customers have higher churn risk). Tenure's raw continuous values might introduce noise or require more complex modeling, so the binned version is 
more efficient and reduces overfitting while preserving the core information.
Part 2: Features to Consider Removing Due to Low Variance or Irrelevance
Before statistical tests (e.g., variance inflation factor or chi-square), I might remove features based on common sense: low variance means the feature doesn't vary much across samples, 
making it uninformative for prediction, while irrelevance means it lacks logical ties to churn behavior. Here are two such features from the original or engineered set, with reasoning.

Feature: CustomerID (Original)

Reasoning: This is an arbitrary identifier with no predictive value—it's unique per customer and doesn't relate to churn behavior (e.g., a higher ID doesn't indicate higher churn risk). 
It has high variance by design (unique values), but that's irrelevant for modeling. Removing it simplifies the dataset without loss, as it's not a true feature.
Feature: Senior_Family_Support (Engineered)

Reasoning: In many churn datasets, SeniorCitizen status is already low-variance (e.g., in the provided data, only 3 out of 15 are seniors or "Yes"), and combining it with Partner/Dependents
 might result in even lower variance if most customers aren't seniors with family. Based on common sense, this feature could be irrelevant if seniors are a small, non-representative group,
  and family support doesn't strongly influence telecom churn (unlike in contexts like healthcare). It might not add unique value beyond SeniorCitizen alone, so I'd remove it to avoid noise
   from sparse data.
"""

"""
Real-World Application

Feature Engineering and Selection are indispensable techniques across nearly all industries that leverage data for predictive modeling. Their application often dictates the success or failure of a machine learning project.

Consider a large telecommunications company facing a significant challenge with customer churn, much like our case study. They might begin with basic customer data: billing information, service usage, contract details, and demographic profiles.

    Initial Data (Raw): MonthlyCharges, DataUsageGB, CallMinutes, ContractType, CustomerAge, ServiceStartDate.
    Feature Engineering:
        They might engineer CustomerTenureInMonths from ServiceStartDate and current date.
        AverageMonthlySpendLast3Months by aggregating MonthlyCharges over recent periods (if transactional data is available).
        ChurnRiskScoreFromSupportCalls by combining NumberOfSupportTicketsLastMonth with AverageResolutionTimeHours (e.g., NumberOfSupportTicketsLastMonth * AverageResolutionTimeHours, or a more complex domain-driven score).
        HasFiberOpticService (binary) from their InternetServiceType.
        IsLongTermContractCustomer (binary) from ContractType.
        DataToCallRatio from DataUsageGB and CallMinutes to understand usage patterns.
    Feature Selection:
        After engineering 50+ new features, they would analyze correlations. For example, TotalDataUsedSinceSignup might be highly correlated with CustomerTenureInMonths. They might decide to keep CustomerTenureInMonths as it's a simpler, more direct measure, or AverageMonthlyDataUsage if it's more representative of current behavior.
        They might find that Gender has a very low statistical correlation with churn, suggesting it's not a strong predictor, and thus remove it.
        Using a tree-based model (an embedded method) they might discover that AverageMonthlySpendLast3Months, ChurnRiskScoreFromSupportCalls, and IsLongTermContractCustomer are among the top 5 most important features for predicting churn, while other engineered features contribute very little.

By effectively performing feature engineering and selection, the telecom company can build a highly accurate churn prediction model. This model then allows them to proactively identify
 customers at high risk of churning and implement targeted retention campaigns (e.g., offering discounts to long-term customers with high churn risk, or specialized tech support to those 
 with high support call scores). Without these steps, their model might be overwhelmed by irrelevant features, miss crucial hidden patterns, and ultimately fail to accurately predict churn,
leading to significant revenue loss.

SOLUTIONs:

Real-World Application: Feature Engineering and Selection in Banking for Loan Default Prediction
Feature Engineering and Selection are critical in industries like banking, where predictive models can prevent financial losses by identifying high-risk borrowers early. Consider 
a major bank grappling with rising loan defaults amid economic uncertainty. They start with basic applicant data: credit score, annual income, loan amount, loan term, employment status, 
age, and application date.

Initial Data (Raw): CreditScore, AnnualIncome, LoanAmount, LoanTermMonths, EmploymentStatus, ApplicantAge, ApplicationDate.
Feature Engineering:
They might engineer ApplicantTenureInYears from ApplicationDate and current date to capture how long the applicant has been a customer.
DebtToIncomeRatio by dividing LoanAmount by AnnualIncome (normalized for loan term if needed) to assess affordability.
CreditUtilizationScore by combining CreditScore with a derived metric like (LoanAmount / (AnnualIncome * LoanTermMonths)) to gauge overall credit strain.
IsStableEmployment (binary) from EmploymentStatus (e.g., "Full-time" or "Self-employed" for 2+ years as stable).
IsHighRiskAgeGroup (binary) from ApplicantAge (e.g., flagging applicants under 25 or over 65 as higher risk based on historical data).
IncomeStabilityIndex from AnnualIncome trends (if historical data is available, e.g., average income over the last 3 years divided by current income).
Feature Selection:
After engineering 40+ new features, they would check correlations. For instance, LoanAmount might be highly correlated with DebtToIncomeRatio. They might keep DebtToIncomeRatio as 
it's a standardized risk indicator, removing LoanAmount to avoid redundancy and multicollinearity.
They might find that ApplicantAge has low variance (e.g., most applicants are 25-55) and weak correlation with defaults, suggesting irrelevance, and thus remove it.
Using a wrapper method like recursive feature elimination (RFE) with a logistic regression model, they might identify DebtToIncomeRatio, CreditUtilizationScore, and IsStableEmployment 
as top predictors, while features like IncomeStabilityIndex (if data is sparse) contribute minimally.
By mastering feature engineering and selection, the bank can develop a robust loan default prediction model. This enables proactive measures like offering lower interest rates to stable, 
low-risk applicants or denying loans to high-risk ones, reducing default rates and bad debt. Without these techniques, the model could be cluttered with noise, overlook key risk factors 
like income strain, and result in poor decisions, exacerbating financial losses during downturns.
"""