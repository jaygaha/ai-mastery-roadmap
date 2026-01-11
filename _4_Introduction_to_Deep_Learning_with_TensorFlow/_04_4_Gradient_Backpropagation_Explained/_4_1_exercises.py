import time

def print_header(title):
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50 + "\n")

def exercise_1_gradient_descent_step():
    print_header("Exercise 1: Gradient Descent Step")
    print("Loss Function: L(w) = (w - 5)^2")
    print("Derivative: dL/dw = 2 * (w - 5)")
    
    w = 8
    learning_rate = 0.1
    
    print(f"\nInitial Weight (w): {w}")
    print(f"Learning Rate: {learning_rate}")
    
    # Step 1 Calculation
    gradient = 2 * (w - 5)
    w_new = w - (learning_rate * gradient)
    
    print(f"\n[Step 1]")
    print(f"Gradient at w={w}: 2 * ({w} - 5) = {gradient}")
    print(f"Update: {w} - ({learning_rate} * {gradient}) = {w_new}")
    
    # Step 2 Calculation
    w_step2 = w_new
    gradient_step2 = 2 * (w_step2 - 5)
    w_final = w_step2 - (learning_rate * gradient_step2)
    
    print(f"\n[Step 2]")
    print(f"Gradient at w={w_step2}: 2 * ({w_step2} - 5) = {gradient_step2}")
    print(f"Update: {w_step2} - ({learning_rate} * {gradient_step2}) = {w_final}")
    
    print(f"\nNotice how the weight is getting closer to 5 (the minimum).")

def exercise_2_chain_rule():
    print_header("Exercise 2: Chain Rule Logic")
    print("Scenario: Input (x) -> Node A (a) -> Loss (L)")
    print("Equations: a = 3x,  L = a^2")
    print("Goal: Find dL/dx")
    
    print("\nAnalytical Solution:")
    print("1. dL/da = derivative of a^2 with respect to a = 2a")
    print("2. da/dx = derivative of 3x with respect to x  = 3")
    print("3. Chain Rule: dL/dx = (dL/da) * (da/dx) = (2a) * 3 = 6a")
    print("   Since a = 3x, dL/dx = 6 * (3x) = 18x")
    
    print("\nLet's verify with a number. Let x = 2.")
    x = 2
    a = 3 * x
    L = a ** 2
    
    # Numerical derivative (slope) check
    epsilon = 0.0001
    x_bump = x + epsilon
    a_bump = 3 * x_bump
    L_bump = a_bump ** 2
    
    rise = L_bump - L
    run = epsilon
    approx_derivative = rise / run
    
    calculated_derivative = 18 * x
    
    print(f"If x={x}:")
    print(f"- Calculated Derivative formula (18x): {calculated_derivative}")
    print(f"- Approximate Derivative (using slight bump): {approx_derivative:.4f}")
    print("They match!")

def exercise_3_learning_rate_demo():
    print_header("Exercise 3: Learning Rate Intuition")
    print("Objective: Find minimum of L(w) = w^2 using Gradient Descent")
    print("Target Minimum: w = 0")
    
    current_w = 10.0
    print(f"Starting Weight: {current_w}")
    
    lr_input = input("\nEnter a learning rate (Try 0.1 for good, 1.1 for bad): ")
    try:
        learning_rate = float(lr_input)
    except ValueError:
        print("Invalid input, defaulting to 0.1")
        learning_rate = 0.1
        
    print(f"\nSimulating 5 steps with Learning Rate: {learning_rate}...")
    
    for i in range(5):
        gradient = 2 * current_w
        update = learning_rate * gradient
        current_w = current_w - update
        print(f"Step {i+1}: w is now {current_w:.4f} (Gradient was {gradient:.2f})")
        time.sleep(0.5)
        
    if abs(current_w) < 1.0:
        print("\nResult: Converging nicely! Good learning rate.")
    elif abs(current_w) > 20:
        print("\nResult: Diverging (Exploding)! Learning rate too high.")
    else:
        print("\nResult: Moving, but maybe too slow or oscillating.")

if __name__ == "__main__":
    while True:
        print("\n--- Gradient & Backprop Exercises ---")
        print("1. Run Exercise 1 (Gradient Descent Step)")
        print("2. Run Exercise 2 (Chain Rule)")
        print("3. Run Exercise 3 (Learning Rate Demo)")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ")
        
        if choice == '1':
            exercise_1_gradient_descent_step()
        elif choice == '2':
            exercise_2_chain_rule()
        elif choice == '3':
            exercise_3_learning_rate_demo()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")
        
        input("\nPress Enter to return to menu...")
