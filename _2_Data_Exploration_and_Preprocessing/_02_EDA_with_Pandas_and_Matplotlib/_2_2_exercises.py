import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('customer_churn.csv')

"""
1. Explore Senior Citizen Distribution:

    Create a bar chart showing the distribution of SeniorCitizen (0 for No, 1 for Yes).
    What insights can you gather about the proportion of senior citizens in the dataset?
"""

senior_counts = df['SeniorCitizen'].value_counts()
plt.figure(figsize=(6, 4))
senior_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribution of Senior Citizen')
plt.xlabel('Senior Citizen (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

"""
Q: What insights can you gather about the proportion of senior citizens in the dataset?

A: Insights: In this small dataset, 4 out of 20 customers are senior citizens (20%), indicating a minority proportion. 
This suggests the dataset is skewed towards non-senior citizens, which may not represent the broader population.
"""

"""
2. Analyze Partner and Dependents:

    Use value_counts() to inspect the distributions of Partner and Dependents columns.
    Calculate the average Tenure for customers with a Partner vs. those without.
    Visualize this comparison using a bar chart or box plot. What does this suggest about the relationship between having a partner and customer loyalty?
"""

# Use value_counts() to inspect the distributions of Partner and Dependents columns.
print("Partner distribution:")
print(df['Partner'].value_counts())
print("\nDependents distribution:")
print(df['Dependents'].value_counts())

#Calculate the average Tenure for customers with a Partner vs. those without.
avg_tenure_partner = df.groupby('Partner')['Tenure'].mean()
print("\nAverage Tenure by Partner:")
print(avg_tenure_partner)

# Visualize this comparison using a bar chart or box plot.
plt.figure(figsize=(8, 6))
avg_tenure_partner.plot(kind='bar', color=['green', 'red'])
plt.title('Average Tenure by Partner')
plt.xlabel('Partner (0=No, 1=Yes)')
plt.ylabel('Average Tenure (months)')
plt.xticks(rotation=0)
plt.show()

"""
3. Investigate PaymentMethod:

    Generate a bar chart to visualize the distribution of different PaymentMethod types.
    Calculate the churn rate for each PaymentMethod. Which payment method has the highest churn rate? Use the Churn_numeric column you created.
    Present the churn rates in a bar chart.
"""

# Generate a bar chart to visualize the distribution of different PaymentMethod types.
payment_counts = df['PaymentMethod'].value_counts()
plt.figure(figsize=(12, 6))
payment_counts.plot(kind='bar', color='purple')
plt.title('Distribution of Payment Methods')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Create Churn_numeric column as requested
df['Churn_numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Calculate the churn rate for each PaymentMethod.
churn_rates = df.groupby('PaymentMethod')['Churn_numeric'].mean()
print("\nChurn rates by Payment Method:")
print(churn_rates)

# Present the churn rates in a bar chart.
plt.figure(figsize=(8, 4))
churn_rates.plot(kind='bar', color='cyan')
plt.title('Churn Rates by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Churn Rate')
plt.xticks(rotation=45)
plt.show()

"""
4. Visualize MonthlyCharges and Tenure with Churn Status:

    Create two separate histograms: one for MonthlyCharges for customers who churned and another for customers who did not churn. Place them side-by-side or on top of each other for easy comparison (you can use plt.subplot).
    Repeat the above for Tenure.
    What differences do you observe in the distributions of MonthlyCharges and Tenure between churned and non-churned customers?
"""

# Histograms for MonthlyCharges: churned vs. not churned, side-by-side.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MonthlyCharges histograms
axes[0].hist(df[df['Churn'] == 'Yes']['MonthlyCharges'], bins=10, alpha=0.7, label='Churned', color='red')
axes[0].hist(df[df['Churn'] == 'No']['MonthlyCharges'], bins=10, alpha=0.7, label='Not Churned', color='blue')
axes[0].set_title('MonthlyCharges Distribution by Churn Status')
axes[0].set_xlabel('Monthly Charges')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Tenure histograms
axes[1].hist(df[df['Churn'] == 'Yes']['Tenure'], bins=10, alpha=0.7, label='Churned', color='red')
axes[1].hist(df[df['Churn'] == 'No']['Tenure'], bins=10, alpha=0.7, label='Not Churned', color='blue')
axes[1].set_title('Tenure Distribution by Churn Status')
axes[1].set_xlabel('Tenure (Months)')
axes[1].set_ylabel('Frequency')
axes[1].legend()
plt.tight_layout()
plt.show()

"""
Q: What differences do you observe in the distributions of MonthlyCharges and Tenure between churned and non-churned customers?
A: Observations: Churned customers tend to have higher MonthlyCharges (peaking around 70-100) compared to non-churned (broader spread, lower peaks). 
For Tenure, churned customers have shorter tenures (mostly under 20 months), while non-churned have longer tenures (up to 70 months), 
indicating that newer customers with higher charges are more likely to churn.
"""