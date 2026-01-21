import matplotlib.pyplot as plt

# Assume 'history' is the object returned by model.fit() in Keras/TensorFlow
# For demonstration purposes, let's create a dummy history object
class DummyHistory:
    def __init__(self):
        self.history = {
            'loss': [0.8, 0.6, 0.4, 0.3, 0.25, 0.23, 0.22, 0.21, 0.20, 0.19],
            'val_loss': [0.85, 0.65, 0.45, 0.38, 0.35, 0.37, 0.40, 0.43, 0.46, 0.49],
            'accuracy': [0.5, 0.65, 0.78, 0.85, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94],
            'val_accuracy': [0.52, 0.68, 0.80, 0.83, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77]
        }

history = DummyHistory()

plt.figure(figsize=(12, 5))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()