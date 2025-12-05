# Data Cleaning: Handling Missing Values, Outliers, and Inconsistencies

Data is the lifeblood of any AI or machine learning model. Just as a chef needs fresh, high-quality ingredients to prepare a gourmet meal, an AI model requires clean, well-structured data to make accurate and reliable predictions. In the real world, however, data rarely arrives in a pristine state. It's often marred by missing entries, extreme values that skew results, and inconsistencies that can mislead algorithms. These "messy" data issues can severely degrade model performance, resulting in flawed insights, inaccurate predictions, and ultimately, poor business decisions. Therefore, mastering data cleaning techniques is not just a best practice—it's a fundamental skill for anyone working in AI and data science. This lesson will teach you essential methods for identifying and resolving common data quality issues, ensuring your data is ready for analysis and model training.

## Understanding Common Data Quality Issues

Before we discuss cleaning techniques, it's important to understand the types of issues you'll encounter in real-world datasets. Data quality problems can be broadly categorized into three types: missing values, outliers, and inconsistencies. Each issue presents unique challenges and requires a specific resolution strategy. Recognizing these issues is the first step toward developing robust, reliable AI systems.

### Missing Values

Missing values, often represented as `NaN` (Not a Number), `None`, `null`, or even an empty string, occur when no data is stored for a particular observation in a specific feature. They are a ubiquitous problem in data analysis and can arise for various reasons:

- **Data Collection Errors:** A sensor malfunctioned, a human failed to record information, or a survey question was left unanswered.
- **Data Entry Errors:** Mistakes during manual data input, leading to blanks or incorrect markers for missingness.
- **Data Integration Issues:** Merging datasets where some fields are not present in all sources.
- **Privacy Concerns:** Certain sensitive information might be intentionally omitted.

**Impact on Models:** Most machine learning algorithms cannot handle missing values directly. They will either throw an error, produce biased results, or make incorrect assumptions about the data. For example, calculating the average income of customers will be inaccurate if a significant portion of income values are missing.

- **Real-world Example 1 (Healthcare):** In a patient record dataset, some entries might have missing blood pressure readings because the patient refused the measurement or the equipment was faulty. If you're building a model to predict heart disease, these missing values could significantly impact the model's ability to learn relationships between blood pressure and disease.
- **Real-world Example 2 (E-commerce):** A customer feedback dataset might have missing values for "purchase amount" if some customers provided feedback without making a recent purchase, or if the system failed to log the purchase amount for that specific feedback entry. A recommendation engine might struggle to provide relevant suggestions without complete purchase history.
- **Hypothetical Scenario:** Imagine a dataset of student performance where some students missed a test. Their scores would be marked as missing. If you just remove these students, you might lose valuable information about other factors affecting performance. If you replace them with an arbitrary value like zero, you might incorrectly depress the average score for that test.

### Outliers

Outliers are data points that significantly deviate from other observations in a dataset. They are values that lie an abnormal distance from other values. Outliers can represent true anomalies, measurement errors, or even data entry mistakes.

- **Data Entry/Measurement Errors:** A human typing 100000 instead of 1000, or a sensor recording a temporary spike due to interference.
- **Natural Variation:** A truly exceptional event or observation (e.g., a multi-million dollar sale in a dataset of typical product sales).
- **Intentional Anomalies:** Fraudulent transactions, cyber-attacks, or rare diseases.

**Impact on Models:** Outliers can disproportionately influence statistical calculations (like means and standard deviations) and model training. Regression models, for instance, are highly sensitive to outliers, as they try to minimize the squared errors, causing the model to bend towards these extreme points. This can lead to models that don't generalize well to new, typical data.

- **Real-world Example 1 (Financial Transactions):** In a credit card transaction dataset, most transactions are small purchases, but a few could be extremely large, representing a car purchase or a fraudulent activity. If a fraud detection model is trained without handling these outliers, it might mistakenly flag legitimate large purchases as fraudulent or miss true fraud patterns if the outliers distort the normal distribution.
- **Real-world Example 2 (Manufacturing Quality Control):** A dataset tracking the thickness of manufactured parts. Most parts fall within a narrow range, but a few might have significantly higher or lower thickness due to a machine malfunction. If not addressed, a predictive maintenance model might fail to accurately predict when machinery needs servicing.
- **Hypothetical Scenario:** Consider a dataset of employee salaries. Most salaries fall within a reasonable range, but the CEO's salary might be an extreme outlier. If you calculate the average salary to represent the "typical" employee income without handling this outlier, the average would be heavily inflated and not representative of the majority of employees.

### Inconsistencies

Data inconsistencies refer to errors that arise from non-uniformity in data representation, contradictory information, or violations of data integrity rules. These often stem from human error, different data sources, or a lack of strict data validation.

- **Data Type Mismatches:** A column intended for numerical values (e.g., age) contains text, or a date column is stored as a generic string.
- **Structural Errors:** Typos, irregular capitalization, or extra spaces in categorical fields (e.g., "Male", "male", " MALE " for gender).
- **Format Errors:** Dates in mixed formats (e.g., "YYYY-MM-DD", "MM/DD/YY"), phone numbers with different separators.
- **Logical Errors/Violations of Business Rules:** An age recorded as 200, a negative price, or a customer's subscription end date occurring before their start date.

**Impact on Models:** Inconsistencies can prevent data from being processed correctly, lead to incorrect groupings, and cause models to misinterpret relationships. For example, if "Male" and "male" are treated as distinct categories, a model won't accurately count or learn from the true "Male" population.

- **Real-world Example 1 (Customer Relationship Management - CRM):** A CRM system might have customer names entered in various formats ("John Doe", "Doe, John", "john doe"). If you're trying to deduplicate customer records or personalize communications, these inconsistencies will make it difficult to identify unique individuals.
- **Real-world Example 2 (Product Catalog):** An online store's product catalog might have inconsistent product categories like "Electronics", "electronics", "ELECTRONICS", and "Elec.". An inventory management system or a recommendation engine would struggle to correctly group and recommend products without standardizing these categories.
- **Hypothetical Scenario:** Imagine a dataset of movie ratings where some ratings are on a 1-5 scale, others are 1-10, and some are descriptive (e.g., "good," "bad"). Without standardizing these, it's impossible to calculate an average rating or train a model to predict user satisfaction. Similarly, a column for "country" might contain "USA", "United States", and "U.S.", which need to be unified.

## Handling Missing Values

Dealing with missing data is a critical step in data cleaning. The approach you choose depends heavily on the nature of the missingness, the amount of data missing, and the potential impact on your analysis. We'll explore identification and two primary strategies: **deletion** and **imputation**.

### Identifying Missing Values

The first step is always to identify *where* and *how much* missing data exists. Pandas provides excellent tools for this.

Let's use our `Customer Churn Prediction Case Study` data. We'll start by loading a sample dataset.
```python
import pandas as pd
import numpy as np

# Create a sample DataFrame resembling the Customer Churn data
# with some deliberate missing values for demonstration
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'SeniorCitizen': [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'Partner': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No'],
    'Dependents': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Tenure': [24, 1, 10, 50, 8, 72, 5, 30, 12, 6],
    'MonthlyCharges': [70.0, 29.85, 104.80, 80.0, 99.65, 110.0, np.nan, 75.0, 90.0, 60.0],
    'TotalCharges': [1676.0, 29.85, np.nan, 3950.0, 800.0, 7933.0, 350.0, 2250.0, np.nan, 360.0],
    'Contract': ['Month-to-month', 'Month-to-month', 'Month-to-month', 'Two year', 'Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Month-to-month', 'Month-to-month'],
    'Churn': ['No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}
df_churn = pd.DataFrame(data)

print("Initial DataFrame head:")
print(df_churn.head())
print("\nDataFrame Info:")
df_churn.info()
```

The `df.info()` method is excellent for getting a quick summary of your DataFrame, including non-null counts, which can indirectly reveal missing values. For direct identification, `isnull()` is key.
```python
# Check for missing values across the entire DataFrame
print("\nMissing values in the DataFrame:")
print(df_churn.isnull())

# Count missing values per column
print("\nNumber of missing values per column:")
print(df_churn.isnull().sum())

# Calculate percentage of missing values per column
print("\nPercentage of missing values per column:")
print((df_churn.isnull().sum() / len(df_churn)) * 100)
```

**Explanation:**

- `df_churn.isnull()`: Returns a DataFrame of boolean values, where `True` indicates a missing value.
- `df_churn.isnull().sum()`: Sums the `True` values for each column, giving the total count of missing values per column.
- `(df_churn.isnull().sum() / len(df_churn)) * 100`: Calculates the percentage of missing values, which is often more insightful than just the raw count.

From our sample data, `MonthlyCharges` has 1 missing value (10%) and `TotalCharges` has 2 missing values (20%).

## Strategies for Handling Missing Values

Once identified, you need to decide how to handle them. The choice between **deletion** and **imputation** is crucial.

1. **Deletion (Dropping Rows or Columns)**
    
    * **Row-wise Deletion** (`dropna()`): This method removes entire rows that contain at least one missing value.

        **When to use:**
        - **Small amount of missing data:** If only a tiny fraction of your rows have missing values, dropping them might be acceptable and won't significantly reduce your dataset size. A common threshold is less than 5% of rows.
        - **Missingness is random:** If the reason for missingness is not related to the underlying data patterns (Missing Completely At Random - MCAR).
        - **Irrelevant data:** If the rows with missing data are not critical for your analysis.

        **When to be cautious:**
        - **Large amount of missing data:** You risk losing a significant portion of your dataset, potentially leading to biased results or loss of valuable information.
        - **Non-random missingness:** If missingness is related to the value itself (e.g., people with low income are less likely to report it), dropping rows can introduce bias.
        ```python
        # Create a copy to demonstrate deletion without altering the original DataFrame
        df_churn_dropped_rows = df_churn.copy()
        print("\nDataFrame before dropping rows (shape):", df_churn_dropped_rows.shape)

        # Drop rows with any missing values
        df_churn_dropped_rows.dropna(inplace=True)
        print("\nDataFrame after dropping rows with any missing values (shape):", df_churn_dropped_rows.shape)
        print("Missing values after dropping rows:")
        print(df_churn_dropped_rows.isnull().sum())
        ```

        **Explanation:**
        - `dropna()`: By default, removes rows (`axis=0`) that contain at least one `NaN`.
        - `inplace=True`: Modifies the DataFrame directly. If `inplace=False` (default), it returns a new DataFrame.
        - Notice how the number of rows decreased. In our example, 3 rows were dropped because one had a missing `MonthlyCharges` and two had missing `TotalCharges`.

    * **Column-wise Deletion** (`dropna(axis=1)`): This method removes entire columns that contain missing values.

        **When to use:**
        - **Very high percentage of missing values in a column:** If a column is almost entirely empty (e.g., >70-80% missing), it might not provide useful information and could be dropped.
        - **Irrelevant column:** If the column is not crucial for your analysis.

        **When to be cautious:**
        
        - **Losing potentially important features:** Even with some missing data, a column might still contain valuable predictive power.
        ```python
        # Create another copy for column dropping
        df_churn_dropped_cols = df_churn.copy()
        print("\nDataFrame before dropping columns (shape):", df_churn_dropped_cols.shape)

        # Drop columns with any missing values
        # We'll demonstrate this on a DataFrame where a column has many NaNs
        # For our current small example, TotalCharges is 20% missing, which might be too much to drop in a real scenario
        # Let's create a scenario where a column is mostly empty
        df_churn_dropped_cols['NewFeature_ManyNaNs'] = [1, np.nan, 3, np.nan, 5, np.nan, 7, np.nan, 9, np.nan]
        print("\nDataFrame with an additional column 'NewFeature_ManyNaNs':")
        print(df_churn_dropped_cols.isnull().sum())

        df_churn_dropped_cols.dropna(axis=1, inplace=True)
        print("\nDataFrame after dropping columns with any missing values (shape):", df_churn_dropped_cols.shape)
        print("Columns after dropping:", df_churn_dropped_cols.columns.tolist())
        ```

        **Explanation:**
        - `dropna(axis=1)`: Removes columns (`axis=1`) that contain at least one `NaN`.
        - In our example, `MonthlyCharges`, `TotalCharges`, and `NewFeature_ManyNaNs` would be dropped. This highlights why column-wise deletion is often too aggressive unless a column is almost entirely empty.
        
2. **Imputation (Filling Missing Values)**

    Imputation involves replacing missing values with substituted values. This is often preferred over deletion when you want to retain as much data as possible.

    **Common Imputation Techniques:**

    - **Mean/Median/Mode Imputation:**

        - **Mean:** Replace missing numerical values with the column's mean. Sensitive to outliers.
        - **Median:** Replace missing numerical values with the column's median. More robust to outliers than the mean.
        - **Mode:** Replace missing categorical or numerical values with the column's most frequent value. Suitable for categorical data or skewed numerical data.
    - **Forward Fill (`ffill()`):** Propagates the last valid observation forward to next valid observation. Useful for time series data.
    - **Backward Fill (`bfill()`):** Propagates the next valid observation backward to previous valid observation. Useful for time series data.

    **When to use:**

    - **Retaining data:** When you cannot afford to lose rows or columns.
    - **Reasonable estimation:** When you believe the imputed value will be a good approximation of the true missing value.

    **When to be cautious:**

    - **Introducing bias:** Imputation can reduce variance and introduce bias if not done carefully.
    - **Distorting relationships:** Replacing missing values with a single constant can distort the correlation between features.

    Let's apply these to our `df_churn` DataFrame.
    ```python
    # Resetting df_churn for imputation examples
    df_churn_imputed = df_churn.copy()

    print("Missing values before imputation:")
    print(df_churn_imputed.isnull().sum())

    # Impute 'MonthlyCharges' with the median
    # The median is generally robust to outliers, making it a good choice for numerical data
    median_monthly_charges = df_churn_imputed['MonthlyCharges'].median()
    df_churn_imputed['MonthlyCharges'].fillna(median_monthly_charges, inplace=True)
    print(f"\n'MonthlyCharges' imputed with median: {median_monthly_charges}")

    # Impute 'TotalCharges' with the mean
    # For demonstration, let's use mean here, though median might be safer for financial data if skewed.
    mean_total_charges = df_churn_imputed['TotalCharges'].mean()
    df_churn_imputed['TotalCharges'].fillna(mean_total_charges, inplace=True)
    print(f"'TotalCharges' imputed with mean: {mean_total_charges}")

    # In a real scenario, you might have categorical columns with missing values.
    # Let's imagine a 'PaymentMethod' column that has a missing value
    df_churn_imputed['PaymentMethod'] = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', np.nan, 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check']
    print("\nMissing values in 'PaymentMethod' before imputation:")
    print(df_churn_imputed['PaymentMethod'].isnull().sum())

    # Impute 'PaymentMethod' with the mode
    mode_payment_method = df_churn_imputed['PaymentMethod'].mode()[0] # .mode() can return multiple if tied, take first
    df_churn_imputed['PaymentMethod'].fillna(mode_payment_method, inplace=True)
    print(f"'PaymentMethod' imputed with mode: {mode_payment_method}")

    print("\nMissing values after imputation:")
    print(df_churn_imputed.isnull().sum())
    print("\nDataFrame head after imputation:")
    print(df_churn_imputed.head())
    ```

    **Explanation:**

    - **`df['Column'].fillna(value, inplace=True)`:** Replaces `NaN` values in the specified column with `value`.
    - **`.median()`, `.mean()`, `.mode()`:** Pandas functions to calculate these statistics. `.mode()` returns a Series, so `[0]` is used to get the first mode in case of ties.

    **Forward/Backward Fill (for sequential data):** While our churn data isn't strictly time-series by rows, if we had sequential data (e.g., customer events logged over time for a single customer), these would be useful.
    ```python
    # Create a small sample for ffill/bfill
    df_sequential = pd.DataFrame({
        'Date': pd.to_datetime(['2025-11-01', '2025-11-02', '2025-11-03', '2025-11-04', '2025-11-05']),
        'Value': [10, np.nan, 12, np.nan, 15]
    })
    print("\nSequential data before ffill:")
    print(df_sequential)

    df_sequential['Value_ffill'] = df_sequential['Value'].ffill()
    print("\nSequential data after ffill:")
    print(df_sequential)

    df_sequential['Value_bfill'] = df_sequential['Value'].bfill()
    print("\nSequential data after bfill:")
    print(df_sequential)
    ```

    **Explanation:**
    - **`ffill()`:** Fills `NaN` values using the previous valid observation.
    - **`bfill()`:** Fills `NaN` values using the next valid observation.

    Choosing the right imputation strategy is crucial. Consider the data distribution (e.g., use median for skewed data), the nature of the feature (categorical vs. numerical), and the potential impact on your model.

## Detecting and Treating Outliers

Outliers can significantly distort analyses and model training. Identifying them is often a combination of visual inspection and statistical methods. Treatment involves either removing them or modifying their values.

### Identifying Outliers

1. **Visual Inspection:**

    Visualizations are excellent for quickly spotting outliers.

    - **Box Plots (`boxplot()`):** Clearly show the distribution of data and highlight potential outliers beyond the "whiskers."
    - **Histograms (`hist()`):** Can show long tails or isolated bars that indicate extreme values.
    - **Scatter Plots (`scatter()`):** Useful for identifying outliers in the context of two variables.

    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Let's use a numerical column from our churn data, e.g., 'Tenure' or 'MonthlyCharges'
    # First, ensure numerical columns are of the correct type (important after imputation)
    df_churn_imputed['MonthlyCharges'] = pd.to_numeric(df_churn_imputed['MonthlyCharges'], errors='coerce')
    df_churn_imputed['TotalCharges'] = pd.to_numeric(df_churn_imputed['TotalCharges'], errors='coerce')
    df_churn_imputed['Tenure'] = pd.to_numeric(df_churn_imputed['Tenure'], errors='coerce')


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_churn_imputed['MonthlyCharges'])
    plt.title('Box Plot of MonthlyCharges')

    plt.subplot(1, 2, 2)
    sns.histplot(df_churn_imputed['MonthlyCharges'], kde=True)
    plt.title('Histogram of MonthlyCharges')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_churn_imputed['Tenure'])
    plt.title('Box Plot of Tenure')

    plt.subplot(1, 2, 2)
    sns.histplot(df_churn_imputed['Tenure'], kde=True)
    plt.title('Histogram of Tenure')

    plt.tight_layout()
    plt.show()

    # For scatter plot, let's plot MonthlyCharges vs. TotalCharges
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Tenure', y='MonthlyCharges', data=df_churn_imputed)
    plt.title('Scatter Plot of Tenure vs. MonthlyCharges')
    plt.show()
    ```

    **Explanation:**

    - Box plots show the median, quartiles, and potential outliers. Points beyond the whiskers are usually considered outliers.
    - Histograms show the frequency distribution. Outliers might appear as small bars far from the main distribution.
    - Scatter plots can reveal data points that are far removed from the general cluster of points.

2. Statistical Methods

    - **Interquartile Range (IQR) Method:** This is a robust method, less sensitive to extreme values than methods relying on the mean and standard deviation.

        - **IQR = Q3 - Q1** (where Q1 is the 25th percentile and Q3 is the 75th percentile).
        - **Lower Bound = Q1 - 1.5 * IQR**
        - **Upper Bound = Q3 + 1.5 * IQR**
        - Any data point below the Lower Bound or above the Upper Bound is considered an outlier.

    - **Z-score Method:** The Z-score measures how many standard deviations an element is from the mean.

        - **Z = (x - mean) / standard deviation**
        - A common threshold for outliers is a Z-score greater than +3 or less than -3 (or +2 / -2 for stricter detection). This method assumes the data is normally distributed.
        ```python
        # Using IQR method for 'MonthlyCharges'
        Q1 = df_churn_imputed['MonthlyCharges'].quantile(0.25)
        Q3 = df_churn_imputed['MonthlyCharges'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"MonthlyCharges - Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"MonthlyCharges - Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")

        outliers_iqr = df_churn_imputed[(df_churn_imputed['MonthlyCharges'] < lower_bound) | (df_churn_imputed['MonthlyCharges'] > upper_bound)]
        print("\nOutliers in 'MonthlyCharges' (IQR method):")
        print(outliers_iqr[['CustomerID', 'MonthlyCharges']])

        # Using Z-score method for 'MonthlyCharges'
        from scipy.stats import zscore

        # Ensure the column is numeric for zscore calculation
        df_churn_imputed['MonthlyCharges_numeric'] = pd.to_numeric(df_churn_imputed['MonthlyCharges'], errors='coerce')
        z_scores = np.abs(zscore(df_churn_imputed['MonthlyCharges_numeric'])) # Calculate absolute Z-scores
        outliers_zscore = df_churn_imputed[z_scores > 2] # Threshold of 2 for demonstration (can be 3 or higher)
        print("\nOutliers in 'MonthlyCharges' (Z-score method, threshold 2):")
        print(outliers_zscore[['CustomerID', 'MonthlyCharges']])

        ```

        **Explanation:**
        
        - `quantile()`: Calculates the specified percentile.
        - `zscore()`: From `scipy.stats`, calculates the Z-score for each value in a Series. We take `np.abs()` because an outlier can be significantly high or low.
        - Using boolean indexing (`df[condition]`) to filter for rows that meet the outlier criteria.

### Treating Outliers

Once identified, outliers need to be treated.

1. **Deletion (Removing Outlier Rows)**

    Similar to missing values, you can remove rows containing outliers.

    **When to use:**

    - **Data entry errors:** If you're confident the outlier is due to an error and not a genuine extreme value.
    - **Small number of outliers:** If removing them doesn't significantly reduce your dataset.
    - **High impact on analysis:** If the outliers severely distort your statistical models.

    **When to be cautious:**

    - **Losing valuable information:** True outliers (e.g., a rare but valid transaction) might contain important insights.
    - **Introducing bias:** If outliers represent a significant subpopulation.

    ```python
    df_churn_no_outliers_deleted = df_churn_imputed.copy()
    print("Shape before outlier deletion:", df_churn_no_outliers_deleted.shape)

    # Removing outliers identified by IQR method
    df_churn_no_outliers_deleted = df_churn_no_outliers_deleted[
        (df_churn_no_outliers_deleted['MonthlyCharges'] >= lower_bound) &
        (df_churn_no_outliers_deleted['MonthlyCharges'] <= upper_bound)
    ]
    print("Shape after outlier deletion (IQR method on MonthlyCharges):", df_churn_no_outliers_deleted.shape)
    ```

2. **Capping (Winsorization)** 

    Capping involves limiting the values of outliers to a specified upper or lower threshold. This replaces extreme values with the boundary values, thus retaining the data points while reducing their influence.

    **When to use:**

    - **Preserving data:** When you don't want to remove data points entirely.
    - **Skewed data:** When outliers are genuine but too extreme for linear models.

    **Common capping methods:**

    * Replace values above `upper_bound` with `upper_bound`.
    * Replace values below `lower_bound` with `lower_bound`.
    * Replace with specific percentiles (e.g., 99th percentile for upper, 1st for lower).

    ```python
    df_churn_capped = df_churn_imputed.copy()

    # Capping 'MonthlyCharges' using IQR bounds
    df_churn_capped['MonthlyCharges'] = np.where(
        df_churn_capped['MonthlyCharges'] > upper_bound,
        upper_bound,
        np.where(
            df_churn_capped['MonthlyCharges'] < lower_bound,
            lower_bound,
            df_churn_capped['MonthlyCharges']
        )
    )

    print("\nOriginal MonthlyCharges statistics:")
    print(df_churn_imputed['MonthlyCharges'].describe())
    print("\nCapped MonthlyCharges statistics (using IQR bounds):")
    print(df_churn_capped['MonthlyCharges'].describe())

    # Verify if any values exceed the bounds after capping
    print(f"Max MonthlyCharges after capping: {df_churn_capped['MonthlyCharges'].max():.2f}")
    print(f"Min MonthlyCharges after capping: {df_churn_capped['MonthlyCharges'].min():.2f}")

    # Visualize the effect of capping
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_churn_imputed['MonthlyCharges'])
    plt.title('MonthlyCharges (Original)')
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_churn_capped['MonthlyCharges'])
    plt.title('MonthlyCharges (Capped)')
    plt.tight_layout()
    plt.show()
    ```

    **Explanation:**

    - `np.where(condition, value_if_true, value_if_false)`: A vectorized function to apply conditional logic.
    The box plot comparison clearly shows how capping compresses the whiskers, indicating that extreme values have been brought within the defined range.

3. **Transformation (Brief Mention)**

    For highly skewed data with outliers, applying mathematical transformations (like logarithm or square root) can reduce the impact of extreme values by compressing the range of data. This topic is more closely related to Feature Engineering, so we'll only mention it here as an alternative for handling outliers by changing their distribution.

## Addressing Data Inconsistencies

Data inconsistencies can be subtle and require careful inspection and systematic correction. They often manifest as incorrect data types, structural errors (typos), or logical errors.

### Identifying Inconsistencies
    
- `df.info()`: Reveals data types. A numerical column stored as object (string) often indicates inconsistencies.
- `df['Column'].unique() / df['Column'].value_counts()`: For categorical columns, these methods help spot variations (e.g., "Male", "male", " MALE").
- `df['Column'].describe()`: For numerical columns, checking min/max values can reveal logical errors (e.g., age 200).
- **Visual inspection**: Histograms and bar charts can sometimes reveal anomalies.

Let's use our `df_churn_imputed` DataFrame (before capping, for a clean slate, or after imputation).
```python
# Let's ensure TotalCharges is numeric, as sometimes missing values can make it 'object' type
df_churn_imputed['TotalCharges'] = pd.to_numeric(df_churn_imputed['TotalCharges'], errors='coerce')
# Check info again
print("DataFrame info after ensuring TotalCharges is numeric:")
df_churn_imputed.info()

print("\nUnique values for 'Gender':")
print(df_churn_imputed['Gender'].unique())

print("\nValue counts for 'Contract':")
print(df_churn_imputed['Contract'].value_counts())

# Introduce an inconsistency for demonstration
df_churn_imputed.loc[0, 'Gender'] = 'MALE '
df_churn_imputed.loc[3, 'Contract'] = 'two year'
df_churn_imputed.loc[5, 'SeniorCitizen'] = 'Yes' # Mix type
df_churn_imputed.loc[8, 'Tenure'] = -5 # Logical error
df_churn_imputed.loc[1, 'MonthlyCharges'] = 1500 # A very high (but not necessarily outlier) value, just for type example

print("\nUnique values for 'Gender' after introducing inconsistency:")
print(df_churn_imputed['Gender'].unique())

print("\nValue counts for 'Contract' after introducing inconsistency:")
print(df_churn_imputed['Contract'].value_counts())

print("\nUnique values for 'SeniorCitizen' after introducing inconsistency:")
print(df_churn_imputed['SeniorCitizen'].unique())

print("\nMin/Max for 'Tenure' after introducing logical error:")
print(f"Min Tenure: {df_churn_imputed['Tenure'].min()}")
print(f"Max Tenure: {df_churn_imputed['Tenure'].max()}")
```

**Explanation:**

- We deliberately introduced some common inconsistencies:

    - `'MALE '` in `Gender` (extra space, different case)
    - `'two year'` in `Contract` (different case)
    - `'Yes'` in `SeniorCitizen` (data type mismatch, should be 0/1)
    - `-5` in `Tenure` (logical error, tenure cannot be negative)

### Correcting Inconsistencies

1. **Data Type Conversion**

    Ensuring columns have the correct data types is fundamental. Pandas offers `astype()`, `to_numeric()`, and `to_datetime()`.
    ```python
    # Correcting 'SeniorCitizen' to numeric (0 or 1)
    # First, map 'Yes' to 1 and 'No' to 0 for consistency, then convert to int
    df_churn_imputed['SeniorCitizen'] = df_churn_imputed['SeniorCitizen'].replace({'Yes': 1, 'No': 0})
    df_churn_imputed['SeniorCitizen'] = pd.to_numeric(df_churn_imputed['SeniorCitizen'], errors='coerce').astype(int)

    # Check the data type and unique values again
    print("\nUnique values for 'SeniorCitizen' after conversion:")
    print(df_churn_imputed['SeniorCitizen'].unique())
    print("Data type of 'SeniorCitizen':", df_churn_imputed['SeniorCitizen'].dtype)
    ```

    **Explanation:**

    - `.replace()`: is used to change specific string values.
    - `pd.to_numeric(errors='coerce')`: This is very useful. If Pandas encounters a value it cannot convert to a number, it will replace it with `NaN` instead of throwing an error. This allows you to handle those `NaN`s later (e.g., by imputation).
    - `.astype(int)`: Converts the column to an integer type.

2. **Structural and Format Errors**

    These often involve string manipulation for categorical columns.

    - **Standardizing Case**: Convert all text to lowercase or uppercase using `.str.lower()` or `.str.upper()`.
    - **Removing Whitespace**: Use `.str.strip()` to remove leading/trailing spaces.
    - **Replacing Typos/Synonyms**: Use `.str.replace()` or `.map()` with a dictionary.
    ```python
    # Standardize 'Gender'
    df_churn_imputed['Gender'] = df_churn_imputed['Gender'].str.strip().str.capitalize()
    print("\nUnique values for 'Gender' after standardization:")
    print(df_churn_imputed['Gender'].unique())

    # Standardize 'Contract'
    df_churn_imputed['Contract'] = df_churn_imputed['Contract'].str.lower().str.replace('two year', 'Two year').str.replace('month-to-month', 'Month-to-month').str.replace('one year', 'One year') # Example of fixing specific cases
    print("\nValue counts for 'Contract' after standardization:")
    print(df_churn_imputed['Contract'].value_counts())
    ```

    **Explanation:**

    - `.str.strip()`: Removes any leading or trailing whitespace.
    - `.str.capitalize()`: Capitalizes the first letter and converts the rest to lowercase.
    - `.str.lower()` / `.str.upper()`: Converts entire string to lower/upper case.
    - `.str.replace()`: Replaces occurrences of a substring. This is effective for fixing specific typos.

3. **Logical Errors (Values outside valid range)**

    These require defining valid ranges and then correcting values that fall outside.
    ```python
    # Correcting 'Tenure' logical error (negative tenure)
    # Tenure cannot be negative. We might replace it with 0 (new customer), median, or mark it as NaN for further handling.
    # Let's replace negative tenure with 0 assuming it indicates a very new customer or an error that should be minimal.
    df_churn_imputed['Tenure'] = np.where(df_churn_imputed['Tenure'] < 0, 0, df_churn_imputed['Tenure'])
    print("\nMin/Max for 'Tenure' after correcting logical error:")
    print(f"Min Tenure: {df_churn_imputed['Tenure'].min()}")
    print(f"Max Tenure: {df_churn_imputed['Tenure'].max()}")

    # Example for MonthlyCharges: Assuming charges cannot be less than 0
    df_churn_imputed['MonthlyCharges'] = np.where(df_churn_imputed['MonthlyCharges'] < 0, 0, df_churn_imputed['MonthlyCharges'])
    print("\nMin MonthlyCharges after correcting logical error (if any):")
    print(f"Min MonthlyCharges: {df_churn_imputed['MonthlyCharges'].min()}")
    ```

    **Explanation:**

    - `np.where()` is again used to apply conditional logic, replacing values that violate the rule.

## Integrated Data Cleaning Workflow: Customer Churn Case Study

Now, let's put all these techniques together in a sequential workflow on our Customer Churn dataset. This demonstrates a typical data cleaning process.

[_3_2_identify_missing_values_case_study.py](./_3_2_identify_missing_values_case_study.py)

**Explanation of Workflow:**

1. **Initial Assessment:** Always start by getting an overview of the data (`info()`, `isnull().sum()`, `unique()`, `describe()`) to identify problems.
2. **Missing Values:** Prioritize handling missing values. Numeric columns are typically imputed with mean/median, and categorical with mode.
3. **Data Type Conversion:** After imputation, ensure columns are in their correct data types, especially if `errors='coerce'` was used.
4. **Inconsistencies:** Standardize string formats (case, whitespace), and correct any logical errors based on domain knowledge.
5. **Outliers:** Detect and treat outliers, often done after missing values are handled to ensure robust statistical calculations (mean, median, IQR).

This sequential approach ensures that each cleaning step builds upon the previous one, leading to a consistently clean dataset.

## Exercise

[_3_3_exercise.py](./_3_3_exercise.py)

## Conclusion

Data cleaning is an indispensable stage in the AI development process, transforming imperfect, raw data into a reliable foundation for analysis and model training. In this lesson, we explored the common causes of messy data, such as missing values, outliers, and inconsistencies, and provided you with practical, modern techniques to address them.

You've learned to:

- **Identify** missing values using `isnull().sum()` and decide between deletion and various imputation strategies like mean, median, or mode.
- **Detect** outliers through visual methods like box plots and statistical methods such as the IQR. You also practiced treating them by deletion or, more commonly, by capping (Winsorization) to preserve data integrity.
- **Correct** inconsistencies ranging from data type mismatches to structural errors like inconsistent capitalization and logical errors that violate domain knowledge.

The Customer Churn Prediction case study offered a practical application of these techniques, showing how a systematic cleaning workflow results in a more reliable dataset. Keep in mind that data cleaning is usually an iterative process that requires domain knowledge and critical thinking to make informed decisions about handling specific data quality issues.

A clean dataset is essential for effective feature engineering and building high-performing models. In our next lesson, "Feature Engineering and Selection for Model Performance," we will use this clean data to create new, more informative features and select the most relevant ones to enhance our model's predictive power. This process guarantees that our AI models are trained using the most accurate representation of reality, resulting in more reliable predictions and insights.
