# Exploratory Data Analysis (EDA) with Pandas and Matplotlib

Exploratory Data Analysis, or EDA, is a really important first step when you're working on any data science or machine learning project. Once you've loaded your raw data, taking the time to really get to know it is key. You want to spot any patterns, find outliers or weird points, and see if your initial ideas make sense. Doing this upfront helps you understand what you're dealing with, which makes everything that comes next — like cleaning up the data, creating new features, or choosing the right model — much easier and more effective.

If you skip doing proper exploratory data analysis, you might end up making decisions without all the facts, building models based on bad data, or overlooking important insights that could really make a difference in how well your model works and how easy it is to understand. In this lesson, I’ll show you the key tools and techniques using popular Python libraries like Pandas for handling data and Matplotlib for visualizing it. This will help you get a clear picture of your data, giving you a strong starting point for building your AI projects.

## The Pillars of Exploratory Data Analysis

EDA is more than just a set of tools. It's a philosophy and an iterative process of examining a dataset to summarize its main characteristics, often using visual methods. Its primary goals are to:

1. **Understand Data Structure:** Understand the organization of the data, including its dimensions (number of rows and columns), column names, and data types. This initial overview ensures that you are working with the data as expected.
    - **Real-world example:** Imagine you're analyzing customer transaction data. EDA helps you quickly confirm if `TransactionID` is unique, `PurchaseAmount` is numerical, and `TransactionDate` is indeed a date/time object, rather than a string.
    - **Hypothetical scenario:** You receive a dataset named `medical_records.csv`. Your first step in EDA is to confirm if it has columns like `PatientID`, `Diagnosis`, `Age`, `Weight`, `BloodPressure`, and whether their data types are appropriate for analysis (e.g., `Age` as integer, `Diagnosis` as string/category).
2. **Identify Patterns and Relationships:** Discovering correlations, trends, and dependencies between different variables. This can reveal which features might be strong predictors for your target variable.
    - **Real-world example:** In a marketing campaign dataset, EDA might reveal that customers who responded positively to a previous campaign also tend to be in a certain age group and have a higher average income. This pattern informs targeting for future campaigns.
    - **Hypothetical scenario:** Analyzing climate data, EDA could show a strong inverse relationship between average temperature and heating energy consumption during winter months, or a direct relationship between rainfall and crop yield in agricultural areas.
3. **Detect Anomalies and Outliers:** Pinpointing unusual data points that might represent errors, rare events, or interesting deviations. These can sometimes be crucial insights or, more often, data entry mistakes that need cleaning.
    - **Real-world example:** In a credit card fraud detection system, an outlier in transaction amount for a specific user might indicate fraudulent activity, or simply a very large legitimate purchase that needs to be accounted for.
    - **Hypothetical scenario:** When examining employee salary data, an outlier representing a salary of $1,000,000 for an entry-level position is likely a data entry error. However, a salary of $500,000 for a CEO might be a legitimate but extreme value.
4. **Check Assumptions:** Verifying any underlying assumptions about the data that might be required for certain statistical models or machine learning algorithms. For instance, checking for normality or linearity.
    - **Real-world example:** Many statistical tests and models (like linear regression) assume that residuals are normally distributed. EDA helps in visually inspecting the distribution of variables to see if this assumption is plausible.
    - **Hypothetical scenario:** If you're planning to use a parametric statistical test that assumes homogeneity of variance between groups, EDA through box plots can visually suggest whether this assumption holds before formal testing.
5. **Inform Subsequent Steps:** The insights gained from EDA directly influence decisions regarding data cleaning (handling missing values, outliers), feature engineering (creating new features), and even the selection of appropriate machine learning models.
    - **Real-world example:** If EDA reveals that a column like `TotalCharges` has many missing values, it might lead to a decision to impute these values, or even drop the column if missingness is too high. If a strong non-linear relationship is observed, it might suggest using a tree-based model instead of a linear model.

## Practical Exploratory Data Analysis with Pandas and Matplotlib

We will use the [Customer Churn Prediction Case Study](https://github.com/jaygaha/ai-mastery-roadmap/tree/main/_1_AI_Foundation_and_Python_Essentials/case_study_customer_churn_prediction/README.md) introduced in Module 1 to demonstrate EDA techniques. For this lesson, we'll assume a dataset `customer_churn.csv` exists with relevant features.

First, let's ensure we have the necessary libraries imported:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # Often useful for numerical operations, though not central to EDA visuals here
```

### Loading and Initial Data Inspection

Before diving deep, it's essential to get a preliminary look at your data.

#### Loading the Data

We'll load our `customer_churn.csv` file into a Pandas DataFrame. Remember from Module 1, Lesson 5, Pandas DataFrames are the workhorse for tabular data in Python.
```python
# Load the dataset
try:
    df = pd.read_csv('customer_churn.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'customer_churn.csv' not found. Please ensure the file is in the correct directory.")
    # Create a dummy DataFrame for demonstration if file not found
    data = {
        # DATA, check _1_1_practical_customer_churn.py for the actual data
    }
    df = pd.DataFrame(data)
    print("Using dummy dataset for demonstration.")

# Display the first few rows to get a quick overview
print("\nFirst 5 rows of the dataset:")
print(df.head())
```

The `df.head()` function is invaluable for a quick visual inspection of the data, showing you column names, initial values, and a glimpse into the data's structure. Similarly, `df.tail()` can show the last few rows.

#### Data Information and Types

Understanding the data types of each column and checking for non-null values is fundamental. The `df.info()` method provides a concise summary.
```python
print("\nDataFrame Info:")
df.info()
```

From `df.info()`, we observe:

- `TotalCharges` is identified as `object` (string) type, which is incorrect for a numerical column. This is a common issue where numbers might be stored as strings due to missing values or special characters. This insight is crucial and will be addressed in future data cleaning lessons. For now, we'll convert it to numeric, handling potential errors.
- We can see which columns have `non-null` values. If a column has fewer `non-null` entries than the total number of entries, it indicates missing values.

Let's convert `TotalCharges` to a numeric type. We use `errors='coerce'` to turn any non-convertible values into `NaN` (Not a Number), which Pandas understands as a missing value.
```python
# Convert TotalCharges to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Re-check info after conversion
print("\nDataFrame Info after converting 'TotalCharges':")
df.info()
```

Now, `TotalCharges` is `float64`, which is correct. The number of non-null values for `TotalCharges` might have decreased, indicating that some original string values couldn't be converted and became `NaN`.

#### Descriptive Statistics

For numerical columns, `df.describe()` provides a statistical summary, including count, mean, standard deviation, min, max, and quartile values. This helps in understanding the distribution, range, and central tendency of the data.
```python
print("\nDescriptive statistics for numerical columns:")
df.describe()
```

Interpretation of df.describe():

- `count`: Number of non-null entries. Can reveal missing data.
- `mean`: Average value.
- `std`: Standard deviation, indicating data dispersion.
- `min, max`: Range of values.
- `25%`, `50%` (median), `75%`: Quartiles, showing the distribution of data points. The median is less sensitive to outliers than the mean.

For categorical columns, `df.describe(include='object')` can give a summary of unique values and their frequencies.
```python
print("\nDescriptive statistics for categorical columns:")
df.describe(include='object')
```

This output shows:

- `count`: Number of non-null entries.
- `unique`: Number of unique categories.
- `top`: The most frequent category.
- `freq`: The frequency of the most frequent category.

#### Unique Values and Counts

For categorical features, it's crucial to examine the unique values and their respective counts to understand the categories present and their distribution.
```python
# Check unique values and their counts for 'Gender'
print("\nValue counts for 'Gender':")
print(df['Gender'].value_counts())

# Check unique values and their counts for 'Contract'
print("\nValue counts for 'Contract':")
print(df['Contract'].value_counts())

# Check number of unique values in 'PaymentMethod'
print("\nNumber of unique payment methods:", df['PaymentMethod'].nunique())
```

These methods help identify potential data entry errors (e.g., 'Male', 'male', 'M') or imbalances in category distribution.

### Visualizing Data Distributions and Relationships with Matplotlib

Matplotlib is a powerful plotting library for Python. While it can create complex visualizations, it's also excellent for generating basic, informative plots essential for EDA.

#### Univariate Analysis (Single Variable)

*Histograms for Numerical Data*

Histograms display the distribution of a single numerical variable by dividing the data into bins and counting the number of observations in each bin. They are excellent for identifying data spread, skewness, and modality.
```python
# Histogram for MonthlyCharges
plt.figure(figsize=(8, 5))
plt.hist(df['MonthlyCharges'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Number of Customers')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Histogram for Tenure
plt.figure(figsize=(8, 5))
plt.hist(df['Tenure'], bins=20, color='lightcoral', edgecolor='black')
plt.title('Distribution of Customer Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')
plt.grid(axis='y', alpha=0.75)
plt.show()
```

**Interpretation:**

- **MonthlyCharges:** The histogram might show a bimodal distribution (two peaks), possibly indicating different pricing tiers or customer segments. It could also be right-skewed, meaning more customers have lower monthly charges.
- **Tenure:** Often shows a high count for very low tenure (new customers) and possibly another peak for very high tenure (loyal customers), with fewer in between, indicating customer retention patterns.

*Box Plots for Numerical Data*

Box plots (or box-and-whisker plots) are effective for visualizing the distribution of numerical data and identifying potential outliers. They show the median, quartiles (25th and 75th percentile), and the range of data, with outliers plotted individually.
```python
# Box plot for MonthlyCharges
plt.figure(figsize=(8, 5))
plt.boxplot(df['MonthlyCharges'])
plt.title('Box Plot of Monthly Charges')
plt.ylabel('Monthly Charges')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Box plot for TotalCharges (after handling NaNs for visualization)
# For simplicity, we drop NaNs for this specific visualization.
# In data cleaning, you'd handle them more robustly.
plt.figure(figsize=(8, 5))
plt.boxplot(df['TotalCharges'].dropna()) # dropna() to handle NaN values for plotting
plt.title('Box Plot of Total Charges')
plt.ylabel('Total Charges')
plt.grid(axis='y', alpha=0.75)
plt.show()
```

**Interpretation:**

- The box represents the interquartile range (IQR), from the 25th to the 75th percentile.
- The line inside the box is the median (50th percentile).
- The "whiskers" extend to the minimum and maximum values within 1.5 times the IQR from the box.
- Points beyond the whiskers are considered outliers.

*Bar Charts for Categorical Data*

Bar charts are used to display the frequency or proportion of observations within different categories of a categorical variable.
```python
# Bar chart for Gender distribution
plt.figure(figsize=(6, 4))
df['Gender'].value_counts().plot(kind='bar', color=['lightseagreen', 'palevioletred'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0) # Keep labels horizontal
plt.grid(axis='y', alpha=0.75)
plt.show()

# Bar chart for Internet Service distribution
plt.figure(figsize=(8, 5))
df['InternetService'].value_counts().plot(kind='bar', color='darkorange')
plt.title('Distribution of Internet Service Types')
plt.xlabel('Internet Service Type')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45) # Rotate labels for readability
plt.grid(axis='y', alpha=0.75)
plt.show()
```

**Interpretation:**

- **Gender:** Shows the count of male vs. female customers, revealing if there's an imbalance.
- **InternetService:** Displays the popularity of different internet service providers (DSL, Fiber optic, No). This can indicate a significant segment of customers not using internet services, which could be important for churn analysis.


#### Bivariate Analysis (Two Variables)

*Scatter Plots for Numerical Relationships*

Scatter plots are used to visualize the relationship between two numerical variables. Each point represents an observation, with its position determined by the values of the two variables.
```python
# Scatter plot between MonthlyCharges and TotalCharges
plt.figure(figsize=(10, 6))
plt.scatter(df['MonthlyCharges'], df['TotalCharges'], alpha=0.6, color='darkblue')
plt.title('Monthly Charges vs. Total Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

**Interpretation:**

- A clear positive linear relationship would show points generally going up from left to right.
- If `TotalCharges` is simply `MonthlyCharges * Tenure`, a strong linear relationship is expected, but variations could suggest different contract terms or price changes over time.
- Any clusters or sparse areas can indicate interesting segments or data artifacts.

*Comparing Numerical by Categorical Variables*

We can use box plots or bar plots to compare a numerical variable across different categories of a categorical variable. This is powerful for understanding how a feature's distribution changes based on a segment.
```python
# Box plot of MonthlyCharges by Churn status
plt.figure(figsize=(8, 6))
df.boxplot(column='MonthlyCharges', by='Churn', grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Monthly Charges by Churn Status')
plt.suptitle('') # Suppress the default matplotlib title for 'by' parameter
plt.xlabel('Churn Status')
plt.ylabel('Monthly Charges')
plt.show()

# Box plot of Tenure by InternetService
plt.figure(figsize=(10, 6))
df.boxplot(column='Tenure', by='InternetService', grid=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Customer Tenure by Internet Service Type')
plt.suptitle('')
plt.xlabel('Internet Service Type')
plt.ylabel('Tenure (Months)')
plt.show()
```

**Interpretation:**

- **MonthlyCharges by Churn:** If customers who churn tend to have higher monthly charges, this plot would show the 'Yes' churn group with a higher median and potentially a different distribution of `MonthlyCharges`. This insight suggests `MonthlyCharges` is an important predictor for churn.
- **Tenure by InternetService:** This can reveal if certain internet service types are associated with longer or shorter customer tenure, which might relate to satisfaction or contract terms.

*Correlation Analysis for Numerical Variables*

Correlation measures the statistical relationship between two variables. Pandas provides `df.corr()` to compute pairwise correlation of columns. For numerical variables, the Pearson correlation coefficient is commonly used, ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation), with 0 indicating no linear correlation.
```python
# Calculate the correlation matrix for numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numerical_cols].corr()

print("\nCorrelation Matrix for Numerical Features:")
print(correlation_matrix)

# Visualize the correlation matrix (basic Matplotlib approach)
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
```

**Interpretation:**

- Values close to 1 or -1 indicate a strong linear relationship.
- Values close to 0 indicate a weak or no linear relationship.
- Diagonal elements are always 1 (a variable is perfectly correlated with itself).
- In our churn case study, we might find `MonthlyCharges` and `TotalCharges` are highly correlated with `Tenure`, as customers with longer tenure naturally accumulate higher total charges and often have stable monthly charges. We would also look for correlations with the `Churn` target (if it were numeric, which we'll address in preprocessing).

#### Grouped Analysis with Pandas

The `groupby()` method in Pandas is extremely powerful for summarizing data by categories. You can apply aggregation functions (like `mean()`, `median()`, `sum()`, `count()`) to numerical columns within each group.
```python
# Calculate the average MonthlyCharges for each Gender
print("\nAverage Monthly Charges by Gender:")
print(df.groupby('Gender')['MonthlyCharges'].mean())

# Calculate the average Tenure for each Contract type
print("\nAverage Tenure by Contract Type:")
print(df.groupby('Contract')['Tenure'].mean())

# Churn rates by Internet Service (requires converting 'Churn' to numeric first)
# For this example, let's map 'Yes' to 1 and 'No' to 0
df['Churn_numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

print("\nChurn Rate by Internet Service:")
print(df.groupby('InternetService')['Churn_numeric'].mean())

# Visualize churn rate by Internet Service
churn_rate_by_internet = df.groupby('InternetService')['Churn_numeric'].mean()
plt.figure(figsize=(8, 5))
churn_rate_by_internet.plot(kind='bar', color='purple')
plt.title('Churn Rate by Internet Service')
plt.xlabel('Internet Service Type')
plt.ylabel('Churn Rate (Proportion)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.show()
```

**Interpretation:**

- **Average Monthly Charges by Gender:** This shows if there's a significant difference in spending between genders.
- **Average Tenure by Contract Type:** Often, customers with 'Two year' contracts have much higher average tenure, as expected, compared to 'Month-to-month' customers. This is a vital insight for churn prediction.
- **Churn Rate by Internet Service:** This might reveal that customers with 'Fiber optic' service have a higher churn rate compared to 'DSL' or 'No internet service', which could point to issues with fiber optic service quality or pricing.

These grouped analyses, particularly when visualized, provide actionable insights that directly inform feature engineering and understanding which aspects of the business drive customer behavior.

## Exercises and Practice Activities

- [Exercises and Practice Activities](./_2_2_exercises.py)

## Conclusion and Next Steps

In this lesson, you gained a foundational understanding of exploratory data analysis (EDA) and its critical role in the machine learning workflow. You learned how to use Pandas to examine data structures, summarize numerical and categorical features, and use Matplotlib to visualize distributions and relationships. These techniques allow you to uncover patterns, identify anomalies, and gain valuable insights to guide your subsequent data processing and model building efforts. We applied these concepts to the customer churn prediction case study, extracting insights about customer demographics and service usage and their potential impact on churn.

The insights gained from EDA are invaluable. For example, identifying missing `TotalCharges` values or strong correlations helps you anticipate challenges and develop strategies for data cleaning and feature engineering. Observing a higher churn rate for specific `InternetService` types or `PaymentMethod` informs which features might be most important for your predictive model.

Building on this understanding, the next lesson, "Data Cleaning: Handling Missing Values, Outliers, and Inconsistencies," will transform these insights into actionable steps for preparing data. You will learn specific techniques for addressing missing values, outliers, and data type inconsistencies that may have been identified during the exploratory data analysis (EDA) process. This will ensure that your data is robust and ready for machine learning.