
"""
Activations & Loss Functions - Solutions
"""

import math

def calculate_mse(actual, predicted):
    """
    Calculate Squared Error for a single prediction.
    """
    error = actual - predicted
    squared_error = error ** 2
    return squared_error

def calculate_categorical_cross_entropy(true_label_index, predicted_probs):
    """
    Calculate Categorical Cross Entropy.
    true_label_index: The index of the correct class (0, 1, or 2)
    predicted_probs: A list of probabilities for each class
    """
    # Get the probability of the correct class
    prob_correct = predicted_probs[true_label_index]
    
    # Formula: -log(probability_of_correct_class)
    loss = -math.log(prob_correct) 
    return loss

def main():
    print("--- Exercise 1: The Right Tool for the Job ---\n")
    print("Scenario: Classifying customers into 'Low', 'Medium', 'High' Risk.")
    print("Question: Which activation function for the output layer?")
    print("\nANSWER: Softmax.")
    print("Reasoning: We have 3 mutually exclusive classes (Multi-Class Classification).")
    print("Sigmoid is strictly for Binary Classification (0 or 1).")
    print("-" * 30 + "\n")

    print("--- Exercise 2: Manually Calculating MSE ---\n")
    actual_temp = 25
    pred_a = 23
    pred_b = 28
    
    # 1. Prediction A Error
    # Error = 25 - 23 = 2
    # Squared Error = 2^2 = 4
    se_a = calculate_mse(actual_temp, pred_a)
    print(f"Prediction A ({pred_a}°C): Squared Error = {se_a}")
    
    # 2. Prediction B Error
    # Error = 25 - 28 = -3
    # Squared Error = (-3)^2 = 9
    se_b = calculate_mse(actual_temp, pred_b)
    print(f"Prediction B ({pred_b}°C): Squared Error = {se_b}")
    
    print("\nConclusion: Prediction B is worse because the squared error (9) is higher than A (4).")
    print("Notice how a difference of 3 leads to more than double the penalty of a difference of 2.")
    print("-" * 30 + "\n")

    print("--- Exercise 3: Calculating Cross-Entropy ---\n")
    print("Scenario: Classifying Fruit [Apple, Banana, Orange]")
    print("True Label: Apple (Index 0)\n")
    
    model_a_probs = [0.9, 0.05, 0.05]
    model_b_probs = [0.3, 0.6, 0.1]
    
    # Model A
    # Probability assigned to Apple: 0.9
    # Loss = -log(0.9)
    loss_a = calculate_categorical_cross_entropy(0, model_a_probs)
    print(f"Model A (0.9 conf): Loss = -log(0.9) = {loss_a:.4f}")
    
    # Model B
    # Probability assigned to Apple: 0.3
    # Loss = -log(0.3)
    loss_b = calculate_categorical_cross_entropy(0, model_b_probs)
    print(f"Model B (0.3 conf): Loss = -log(0.3) = {loss_b:.4f}")
    
    print("\nConclusion: Model B has a MUCH higher loss (1.20 vs 0.10).")
    print("Cross-entropy heavily penalizes the model for being unsure about the correct class.")

if __name__ == "__main__":
    main()
