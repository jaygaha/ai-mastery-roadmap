"""
# Exercise 2: Perceptron Learning Walkthrough

This exercise shows how the weights and bias "correct" themselves when the 
model makes a wrong prediction. We use a single input $x$ to keep the math easy to follow.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

print("\nGiven Information:")
print("   Initial weight: w = 0.3")
print("   Initial bias: b = -0.1")
print("   Learning rate: alpha = 0.1")
print("   Activation: Step function (1 if z >= 0, else 0)")

print("\nLearning Rule Formula:")
print("   1. Predict: y^ = step(w*x + b)")
print("   2. Calculate error: e = y - y^")
print("   3. Update weight: w_new = w_old + alpha*e*x")
print("   4. Update bias: b_new = b_old + alpha*e")

# Initial values
w = 0.3
b = -0.1
alpha = 0.1

# Training examples
training_data = [(1, 1), (0, 0)]

for i, (x, y_true) in enumerate(training_data, 1):
    print(f"TRAINING EXAMPLE {i}: x={x}, y={y_true}")
    
    print(f"\n   Current parameters: w={w:.4f}, b={b:.4f}")
    
    # Step 1: Forward pass
    print(f"\n   Step 1: Calculate prediction")
    z = w*x + b
    print(f"      z = w*x + b = {w:.4f}*{x} + {b:.4f} = {z:.4f}")
    y_pred = 1 if z >= 0 else 0
    print(f"      y^ = step(z) = step({z:.4f}) = {y_pred}")
    
    # Step 2: Calculate error
    error = y_true - y_pred
    print(f"\n   Step 2: Calculate error")
    print(f"      e = y - y^ = {y_true} - {y_pred} = {error}")
    
    if error == 0:
        print(f"      Prediction correct! No update needed.")
    else:
        print(f"      Prediction wrong! Need to update.")
    
    # Step 3: Update weights
    print(f"\n   Step 3: Update weight")
    w_old = w
    w = w + alpha * error * x
    print(f"      w_new = w_old + alpha*e*x")
    print(f"      w_new = {w_old:.4f} + {alpha}*{error}*{x}")
    print(f"      w_new = {w_old:.4f} + {alpha*error*x:.4f}")
    print(f"      w_new = {w:.4f}")
    
    # Step 4: Update bias
    print(f"\n   Step 4: Update bias")
    b_old = b
    b = b + alpha * error
    print(f"      b_new = b_old + alpha*e")
    print(f"      b_new = {b_old:.4f} + {alpha}*{error}")
    print(f"      b_new = {b_old:.4f} + {alpha*error:.4f}")
    print(f"      b_new = {b:.4f}")
    
    print(f"\n   Updated parameters: w={w:.4f}, b={b:.4f}")

print(f"\n{'='*70}")
print(f"FINAL RESULTS AFTER ONE EPOCH")
print(f"{'='*70}")
print(f"\n   Final weight: w = {w:.4f}")
print(f"   Final bias: b = {b:.4f}")