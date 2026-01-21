import matplotlib.pyplot as plt

# 1. Setup Data
history_data = {
    'loss': [0.9, 0.7, 0.5, 0.35, 0.28, 0.22, 0.18, 0.15, 0.13, 0.11],
    'val_loss': [0.95, 0.78, 0.6, 0.48, 0.45, 0.47, 0.50, 0.53, 0.56, 0.59],
    'accuracy': [0.45, 0.60, 0.75, 0.85, 0.89, 0.92, 0.94, 0.95, 0.96, 0.97],
    'val_accuracy': [0.48, 0.62, 0.72, 0.78, 0.80, 0.79, 0.78, 0.77, 0.76, 0.75]
}

# Epochs (1 to 10)
epochs = range(1, len(history_data['loss']) + 1)

# 2. Plotting
plt.figure(figsize=(12, 5))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(epochs, history_data['loss'], 'bo-', label='Training Loss')
plt.plot(epochs, history_data['val_loss'], 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(epochs, history_data['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(epochs, history_data['val_accuracy'], 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

## 3. Analysis: Detective Work
"""
ANALYSIS REPORT: WHERE DID IT GO WRONG?

1.  **The "Happiness" Phase (Epochs 1-4):**
    -   Both Training Loss (Blue) and Validation Loss (Red) are going down.
    -   This means the model is genuinely learning patterns that work for EVERYONE (train data and new data).

2.  **The Turning Point (Epoch 5):**
    -   Look closely at Epoch 5. The Red line (Validation) reaches its lowest point (around 0.45).
    -   After this, the lines start to separate.

3.  **The "Memorization" Phase (Epochs 6-10):**
    -   Training Loss keeps dropping (The model is thinking: "I'm getting smarter!").
    -   BUT Validation Loss starts rising (The test text says: "No, you're just memorizing the answers!").
    -   **Verdict:** The model started OVERFITTING at **Epoch 5**.
    
    **Recommendation:**
    -   We should have stopped the training at Epoch 5.
    -   Or, we need to add "Regularization" (like Dropout or L2) to punish the model for memorizing too much detail.
"""
