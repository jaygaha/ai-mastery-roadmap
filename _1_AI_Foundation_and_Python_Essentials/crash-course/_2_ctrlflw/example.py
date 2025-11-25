# --- Scenario: Analyzing Customer Interaction Logs ---
# Imagine you have a list of recent customer interactions, where each interaction
# is represented as a dictionary.

customer_interactions = [
    {"customer_id": "C001", "event": "login", "timestamp": "2025-11-01 09:00:00", "status": "success"},
    {"customer_id": "C002", "event": "view_product", "timestamp": "2025-11-01 09:05:15", "product_id": "P101"},
    {"customer_id": "C001", "event": "add_to_cart", "timestamp": "2025-11-01 09:10:30", "product_id": "P101", "quantity": 1},
    {"customer_id": "C003", "event": "login", "timestamp": "2025-11-01 09:12:00", "status": "failed"},
    {"customer_id": "C002", "event": "checkout", "timestamp": "2025-11-01 09:20:00", "total_amount": 49.99},
    {"customer_id": "C001", "event": "view_product", "timestamp": "2025-11-01 09:25:00", "product_id": "P102"},
    {"customer_id": "C004", "event": "login", "timestamp": "2025-11-01 09:30:00", "status": "success"},
    {"customer_id": "C003", "event": "login", "timestamp": "2025-11-01 09:35:00", "status": "success"} # Successful login after failure
]

# Task 1: Find all unique customer IDs that interacted.
unique_customer_ids = set()
for interaction in customer_interactions:
    unique_customer_ids.add(interaction["customer_id"])
print(f"Unique customer IDs: {unique_customer_ids}")

# Task 2: Count successful and failed login attempts.
successful_logins = 0
failed_logins = 0

for interaction in customer_interactions:
    if interaction["event"] == "login":
        if interaction["status"] == "success":
            successful_logins += 1
        elif interaction["status"] == "failed":
            failed_logins += 1

print(f"\nSuccessful logins: {successful_logins}")
print(f"Failed logins: {failed_logins}")

# Task 3: List all products added to cart or viewed.
# We'll collect unique product IDs
products_of_interest = set()
for interaction in customer_interactions:
    # Use .get() to safely access 'product_id' as not all interactions have it
    product_id = interaction.get("product_id")
    if product_id:  # Only proceed if product_id exists and is not None
        products_of_interest.add(product_id)

print(f"\nProducts added to cart or viewed: {products_of_interest}")

# Task 4: Find interactions by a specific customer and calculate their total spend if any.
target_customer = "C002"
customer_c002_actions = []
total_spend_c002 = 0.0

for interaction in customer_interactions:
    if interaction["customer_id"] == target_customer:
        customer_c002_actions.append(interaction)
        # Check if this interaction has a 'total_amount' key
        if "total_amount" in interaction:
            total_spend_c002 += interaction["total_amount"]

print(f"\nInteractions for customer {target_customer}:")
for action in customer_c002_actions:
    print(f"- Event: {action['event']}, Timestamp: {action['timestamp']}")
    print(f"Total spend for customer {target_customer}: ¥{total_spend_c002:.2f}")

# Task 5: Create a summary of customer activity for successful logins
# We will create a dictionary where keys are customer IDs and values are lists of their successful login timestamps.
customer_successful_logins_summary = {}

for interaction in customer_interactions:
    if interaction["event"] == "login" and interaction["status"] == "success":
        customer_id = interaction["customer_id"]
        timestamp = interaction["timestamp"]

        # If customer ID not yet a key, initialize with an empty list
        if customer_id not in customer_successful_logins_summary:
            customer_successful_logins_summary[customer_id] = []

        # Append the timestamp to the customer's list
        customer_successful_logins_summary[customer_id].append(interaction)

print("\nCustomer Successful Login Summary:")
for customer_id, login_times in customer_successful_logins_summary.items():
    print(f"  {customer_id}: {login_times}")