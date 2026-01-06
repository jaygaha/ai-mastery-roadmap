"""
# Exercise 3: Linear Separability Analysis

Linear separability is the "straight line test." If you can separate two 
different groups of points with a single ruler, they are linearly separable. 
If you need a curved line or multiple circles, they are not, and a simple 
perceptron will fail.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

print("\nDataset:")
data = np.array([
    [1, 1, 0],
    [1, 2, 0],
    [2, 1, 0],
    [2, 2, 1]
])

df_data = {
    'x1': data[:, 0],
    'x2': data[:, 1],
    'Class (y)': data[:, 2]
}

print("\n   x1  x2  Class")
print("   " + "-"*15)
for i in range(len(data)):
    print(f"   {data[i,0]:.0f}   {data[i,1]:.0f}   {data[i,2]:.0f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Data points
ax1 = axes[0]

class_0 = data[data[:, 2] == 0]
class_1 = data[data[:, 2] == 1]

ax1.scatter(class_0[:, 0], class_0[:, 1], s=300, c='red', marker='o',
           edgecolors='black', linewidth=2, label='Class 0', zorder=5)
ax1.scatter(class_1[:, 0], class_1[:, 1], s=300, c='green', marker='s',
           edgecolors='black', linewidth=2, label='Class 1', zorder=5)

# Annotate points
for i, (x1, x2, y) in enumerate(data):
    ax1.annotate(f'({x1:.0f},{x2:.0f})', (x1, x2), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold')

ax1.set_xlabel('x1', fontsize=12, fontweight='bold')
ax1.set_ylabel('x2', fontsize=12, fontweight='bold')
ax1.set_title('Dataset Visualization', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, 2.5)
ax1.set_ylim(0.5, 2.5)

# Plot 2: Attempting separation
ax2 = axes[1]

ax2.scatter(class_0[:, 0], class_0[:, 1], s=300, c='red', marker='o',
           edgecolors='black', linewidth=2, label='Class 0', zorder=5)
ax2.scatter(class_1[:, 0], class_1[:, 1], s=300, c='green', marker='s',
           edgecolors='black', linewidth=2, label='Class 1', zorder=5)

# Try to draw a separating line
# Best attempt: x1 + x2 = 3.5
x1_sep = np.linspace(0.5, 2.5, 100)
x2_sep = 3.5 - x1_sep
ax2.plot(x1_sep, x2_sep, 'b--', linewidth=2, label='Best attempt line', alpha=0.7)

# Shade regions
ax2.fill_between(x1_sep, x2_sep, 3, alpha=0.2, color='green')
ax2.fill_between(x1_sep, 0, x2_sep, alpha=0.2, color='red')

ax2.set_xlabel('x1', fontsize=12, fontweight='bold')
ax2.set_ylabel('x2', fontsize=12, fontweight='bold')
ax2.set_title('Attempting Linear Separation', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 2.5)
ax2.set_ylim(0.5, 2.5)

plt.tight_layout()
plt.show()

print("\nCan a single perceptron classify this data perfectly?")
print("\nANSWER: YES! This data IS linearly separable.")

print("\nAnalysis:")
print("\n   Class 0 points: (1,1), (1,2), (2,1)")
print("   Class 1 point:  (2,2)")

print("\n   Notice the pattern:")
print("   • All Class 0 points have x1 + x2 < 4")
print("   • Class 1 point has x1 + x2 = 4")

print("\n   A line x1 + x2 = 3.5 can separate them:")
print("   • Below/left of line -> Class 0")
print("   • Above/right of line -> Class 1")

print("\nWhy is this linearly separable?")
print("    We CAN draw a straight line that separates the classes")
print("    All Class 0 points are on one side")
print("    Class 1 point is on the other side")
print("    A single perceptron CAN learn this!")

print("\nPerceptron equation that works:")
print("   Decision boundary: x1 + x2 = 3.5")
print("   Perceptron: y^ = step(x1 + x2 - 3.5)")
print("   Weights: w1=1, w2=1, b=-3.5")

print("\nVerification:")
for x1, x2, y_true in data:
    z = 1*x1 + 1*x2 - 3.5
    y_pred = 1 if z >= 0 else 0
    check = "✓" if y_pred == y_true else "✗"
    print(f"   ({x1:.0f},{x2:.0f}): z={z:.1f}, y^={y_pred}, y={y_true:.0f} {check}")

# Example of NON-linearly separable data
print("Example of NON-Linearly Separable Data (XOR)")

xor_data = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

print("\n   XOR Dataset:")
print("   x₁  x₂  y")
print("   " + "-"*10)
for row in xor_data:
    print(f"   {row[0]:.0f}   {row[1]:.0f}   {row[2]:.0f}")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

xor_class_0 = xor_data[xor_data[:, 2] == 0]
xor_class_1 = xor_data[xor_data[:, 2] == 1]

ax.scatter(xor_class_0[:, 0], xor_class_0[:, 1], s=400, c='red', marker='o',
          edgecolors='black', linewidth=2, label='Class 0')
ax.scatter(xor_class_1[:, 0], xor_class_1[:, 1], s=400, c='green', marker='s',
          edgecolors='black', linewidth=2, label='Class 1')

for x1, x2, y in xor_data:
    ax.annotate(f'({x1:.0f},{x2:.0f})\ny={y:.0f}', (x1, x2), 
               xytext=(10, 10), textcoords='offset points',
               fontsize=10, fontweight='bold')

# Try to draw ANY line - it won't work!
x_line = np.linspace(-0.5, 1.5, 100)
y_line = 0.5 + 0*x_line
ax.plot(x_line, y_line, 'b--', linewidth=2, alpha=0.5, label='No line works!')

ax.set_xlabel('x1', fontsize=12, fontweight='bold')
ax.set_ylabel('x2', fontsize=12, fontweight='bold')
ax.set_title('XOR Problem - NOT Linearly Separable', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.show()

print("\n Why XOR is NOT linearly separable:")
print("   • Classes are arranged in diagonal pattern")
print("   • No single straight line can separate them")
print("   • Would need a neural network with hidden layers!")
