# Conditional Statements: if, elif, else
# -> Conditional statements allow your program to execute different blocks of code based on whether certain conditions
#       are true or false.

#
# If statement
# -> An if statement evaluates a condition. If the condition is True, the indented block of code following the if
#       statement is executed.

print("\nif statement examples")
temperature = 28

if temperature > 25:
    print("It's hot outside!")

humidity = 70
if humidity > 60:
    print("Humidity is high.")

#
# if-else Statement
# -> The if-else statement provides an alternative block of code to execute if the if condition is false.

print("\nif-else statement examples")
score = 85

if score >= 60:
    print("You passed the exam.")
else:
    print("You need to study more.")

#
# if-elif-else Statement
# -> For multiple conditions, you can use elif (short for "else if"). Python will check conditions sequentially and
#   execute the block corresponding to the first True condition. If none of the if or elif conditions are met, the else
#   block (if present) is executed.

print("\nif-elif-else statement examples")
grade = 75

if grade >= 90:
    print("Grade: A")
elif grade >= 80:
    print("Grade: B")
elif grade >= 70:
    print("Grade: C")
elif grade >= 60:
    print("Grade: D")
else:
    print("Grade: F")

# Real-world scenario: Categorizing customer engagement
last_login_days = 15

if last_login_days <= 7:
    print("Customer is highly engaged.")
elif last_login_days <= 30:
    print("Customer is engaged.")
elif last_login_days <= 90:
    print("Customer engagement is low.")
else:
    print("Customer is inactive.")

#
# Logical Operators (and, or, not)
# You can combine multiple conditions using logical operators:
#
#     and: Both conditions must be True.
#     or: At least one condition must be True.
#     not: Reverses the truth value of a condition.

print("\nLogical Operators (and, or, not)")
age = 22
is_student = True

if age >= 18 and is_student:
    print("Eligible for student discount.")

has_premium_account = False
is_logged_in = True

if has_premium_account or is_logged_in:
    print("Access to basic features granted.")

is_maintenance_mode = False
if not is_maintenance_mode:
    print("System is operational.")

# Hypothetical scenario: Credit score check for a loan
credit_score = 720
income = 55000
debt_to_income_ratio = 0.35

if (credit_score >= 700 and income >= 50000) or debt_to_income_ratio < 0.40:
    print("Loan application approved.")
else:
    print("Loan application denied.")