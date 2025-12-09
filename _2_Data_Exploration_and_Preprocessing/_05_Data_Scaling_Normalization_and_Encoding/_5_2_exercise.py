import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ==========================================
# Exercise 1: Data Scaling (Min-Max Scaling)
# ==========================================
print("Exercise 1: Min-Max Scaling")
print("-" * 30)

# Scenario:
# Feature 1: Customer Feedback Scores (1 to 10)
# Feature 2: Support Tickets (0 to 1000)

# Create a sample dataset
data_ex1 = {
    'FeedbackScore': [7, 2, 9, 5, 8],
    'SupportTickets': [50, 900, 10, 300, 120]
}
df_ex1 = pd.DataFrame(data_ex1)

print("Original DataFrame (Exercise 1):")
print(df_ex1)
print()

# Task: Apply MinMaxScaler to both features
# Explanation: We use MinMaxScaler to bound both features between 0 and 1,
# preventing the large range of 'SupportTickets' from dominating 'FeedbackScore'.

scaler = MinMaxScaler()
df_ex1_scaled = df_ex1.copy()
df_ex1_scaled[['FeedbackScore', 'SupportTickets']] = scaler.fit_transform(df_ex1[['FeedbackScore', 'SupportTickets']])

print("Scaled DataFrame (Exercise 1):")
print(df_ex1_scaled)
print("-" * 30)
print()


# ==========================================
# Exercise 2: Encoding Categorical Variables
# ==========================================
print("Exercise 2: One-Hot Encoding")
print("-" * 30)

# Scenario:
# Feature: PaymentMethod ('Electronic check', 'Mailed check', 'Credit card (automatic)')

# Create a sample dataset
data_ex2 = {
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Credit card (automatic)', 'Electronic check']
}
df_ex2 = pd.DataFrame(data_ex2)

print("Original DataFrame (Exercise 2):")
print(df_ex2)
print()

# Task: Apply One-Hot Encoding
# Explanation: 'PaymentMethod' is nominal (no inherent order), so One-Hot Encoding is appropriate.

encoder = OneHotEncoder(sparse_output=False)
encoded_payment = encoder.fit_transform(df_ex2[['PaymentMethod']])

# Create a DataFrame for the encoded features
# get_feature_names_out() retrieves the column names created by the encoder
encoded_df = pd.DataFrame(encoded_payment, columns=encoder.get_feature_names_out(['PaymentMethod']))

print("One-Hot Encoded Features (Exercise 2):")
print(encoded_df)
print("-" * 30)
