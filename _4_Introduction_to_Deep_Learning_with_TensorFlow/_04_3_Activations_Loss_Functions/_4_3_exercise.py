
"""
Activations & Loss Functions - Exercises
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
    # matching the natural log (ln) used in most ML libraries
    loss = -math.log(prob_correct) 
    return loss

def main():
    print("--- Exercise 1: The Right Tool for the Job ---\n")
    print("Scenario: Classifying customers into 'Low', 'Medium', 'High' Risk.")
    print("Question: Which activation function for the output layer?")
    input("Press Enter to see the answer...")
    print("\nANSWER: Softmax.")
    print("Why? Because we have 3 mutually exclusive classes. Sigmoid is for binary (2-class) problems.")
    print("-" * 30 + "\n")

    print("--- Exercise 2: Manually Calculating MSE ---\n")
    actual_temp = 25
    pred_a = 23
    pred_b = 28
    
    print(f"Actual: {actual_temp}, Prediction A: {pred_a}, Prediction B: {pred_b}")
    
    print("\n1. Calculate Squared Error for Prediction A:")
    # TODO: Calculate error manually or call the function
    # se_a = ...
    # print(f"Squared Error A: {se_a}")
    
    print("\n2. Calculate Squared Error for Prediction B:")
    # TODO: Calculate error manually or call the function
    # se_b = ...
    # print(f"Squared Error B: {se_b}")
    
    input("\nPress Enter to calculate and reveal answers...")
    
    se_a = calculate_mse(actual_temp, pred_a)
    se_b = calculate_mse(actual_temp, pred_b)
    
    print(f"Squared Error A: {se_a}")
    print(f"Squared Error B: {se_b}")
    
    if se_b > se_a:
        print("\nConclusion: Prediction B is worse (higher error).")
    else:
        print("\nConclusion: Prediction A is worse (higher error).")
    print("-" * 30 + "\n")

    print("--- Exercise 3: Calculating Cross-Entropy ---\n")
    print("Scenario: Classifying Fruit [Apple, Banana, Orange]")
    print("True Label: Apple (Index 0)")
    
    model_a_probs = [0.9, 0.05, 0.05]
    model_b_probs = [0.3, 0.6, 0.1]
    
    print(f"\nModel A Probs: {model_a_probs}")
    print(f"Model B Probs: {model_b_probs}")
    
    print("\n1. Calculate Loss for Model A (High Confidence Correct):")
    # loss_a = ...
    
    print("\n2. Calculate Loss for Model B (Low Confidence Correct):")
    # loss_b = ...
    
    input("\nPress Enter to calculate and reveal answers...")
    
    loss_a = calculate_categorical_cross_entropy(0, model_a_probs)
    loss_b = calculate_categorical_cross_entropy(0, model_b_probs)
    
    print(f"Loss Model A: {loss_a:.4f}")
    print(f"Loss Model B: {loss_b:.4f}")
    
    if loss_b > loss_a:
        print("\nConclusion: Model B has higher loss (it was less confident/wrong).")
    else:
         print("\nConclusion: Model A has higher loss.")

if __name__ == "__main__":
    main()
