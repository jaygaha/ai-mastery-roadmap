from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Re-using the dummy data from the confusion matrix example for y_true
# and y_pred_proba (probabilities)
# Let's create a more realistic scenario where we train a simple model to get probabilities

# Dummy dataset for demonstration (binary classification)
X = np.random.rand(100, 10) # 100 samples, 10 features
y = np.random.randint(0, 2, size=100) # Binary labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple Keras model
model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid') # Sigmoid for binary classification
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0) # verbose=0 to suppress output

# Get predicted probabilities for the positive class (class 1)
y_pred_proba_keras = model.predict(X_test).ravel()

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_keras)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# EDUCATIONAL NOTE:
# An AUC of 0.5 means the model is as good as a coin flip.
# An AUC of 1.0 means the model is perfect.
# An AUC of 0.85 means: "If you pick a random Positive and a random Negative, 
# there is an 85% chance the model scores the Positive higher."

plt.grid(True)
plt.show()