"""
# Exercise 1: Manual Perceptron Calculation

In this exercise, we manually calculate the output of a perceptron with two inputs, 
specific weights, and a bias. This helps visualize how the "Weighted Sum" and 
"Activation Function" work together.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

print("\nGiven Information:")
print("   Weights: w₁ = 0.5, w₂ = -0.2")
print("   Bias: b = 0.1")
print("   Activation: Step function (1 if sum ≥ 0, else 0)")

print("\nFormula:")
print("   Step 1: Calculate weighted sum: z = w₁x₁ + w₂x₂ + b")
print("   Step 2: Apply activation: ŷ = 1 if z ≥ 0, else 0")

# Given values
w1, w2 = 0.5, -0.2
b = 0.1

def perceptron_output(x1, x2, w1, w2, b):
    """Calculate perceptron output"""
    z = w1*x1 + w2*x2 + b
    y_hat = 1 if z >= 0 else 0
    return z, y_hat

# Test cases
test_cases = [
    (1, 0, 'a'),
    (0, 1, 'b'),
    (1, 1, 'c')
]

results = []

for x1, x2, label in test_cases:
    print(f"Case {label}: x₁ = {x1}, x₂ = {x2}")
    
    print(f"\n   Step 1: Calculate weighted sum")
    print(f"      z = w₁×x₁ + w₂×x₂ + b")
    print(f"      z = {w1}×{x1} + {w2}×{x2} + {b}")
    print(f"      z = {w1*x1} + {w2*x2} + {b}")
    
    z, y_hat = perceptron_output(x1, x2, w1, w2, b)
    print(f"      z = {z}")
    
    print(f"\n   Step 2: Apply step function")
    print(f"      Is z ≥ 0? {z} ≥ 0? {z >= 0}")
    print(f"      ŷ = {y_hat}")
    
    print(f"\n   ANSWER: ŷ = {y_hat}")
    
    results.append((x1, x2, z, y_hat))

# Visualization
print("\nVISUALIZATION")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Decision boundary
ax1 = axes[0]


# Plot the points
for x1, x2, z, y_hat in results:
    if y_hat == 1:
        ax1.scatter(x1, x2, s=300, c='green', marker='o', 
                   edgecolors='black', linewidth=2, zorder=5)
    else:
        ax1.scatter(x1, x2, s=300, c='red', marker='x', 
                   linewidth=3, zorder=5)
    ax1.annotate(f'({x1},{x2})\ny^={y_hat}', (x1, x2), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold')

# Plot decision boundary: w₁x₁ + w₂x₂ + b = 0
# Rearrange: x₂ = -(w₁x₁ + b) / w₂
x1_line = np.linspace(-0.5, 1.5, 100)
x2_line = -(w1*x1_line + b) / w2
ax1.plot(x1_line, x2_line, 'b-', linewidth=2, label='Decision Boundary')

# Shade regions
ax1.fill_between(x1_line, x2_line, 2, alpha=0.2, color='green', label='y^=1 region')
ax1.fill_between(x1_line, -1, x2_line, alpha=0.2, color='red', label='y^=0 region')

ax1.set_xlabel('x1', fontsize=12, fontweight='bold')
ax1.set_ylabel('x2', fontsize=12, fontweight='bold')
ax1.set_title('Perceptron Decision Boundary', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-1, 2)

# Plot 2: Weighted sum values
ax2 = axes[1]
labels = ['Case a\n(1,0)', 'Case b\n(0,1)', 'Case c\n(1,1)']
z_values = [r[2] for r in results]
colors_z = ['green' if z >= 0 else 'red' for z in z_values]

bars = ax2.bar(labels, z_values, color=colors_z, alpha=0.7, 
              edgecolor='black', linewidth=2)
ax2.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='Threshold (0)')

for bar, z in zip(bars, z_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height,
            f'z={z:.1f}', ha='center', 
            va='bottom' if height >= 0 else 'top',
            fontsize=11, fontweight='bold')

ax2.set_ylabel('Weighted Sum (z)', fontsize=12, fontweight='bold')
ax2.set_title('Weighted Sum Values', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()