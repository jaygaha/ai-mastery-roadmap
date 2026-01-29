import numpy as np

def simple_rnn_step(input_val, hidden_state, W_hh, W_xh, W_hy, b_h, b_y):
    """
    Performs a single time step of a vanilla RNN.
    
    Args:
        input_val (float): The input at current time step x_t.
        hidden_state (float): The hidden state from previous time step h_{t-1}.
        W_hh (float): Weight for hidden-to-hidden connection.
        W_xh (float): Weight for input-to-hidden connection.
        W_hy (float): Weight for hidden-to-output connection.
        b_h (float): Bias for hidden state.
        b_y (float): Bias for output.
        
    Returns:
        new_hidden_state (float): h_t
        output (float): y_t
    """
    # 1. Calculate new hidden state: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
    # We use tanh to keep values between -1 and 1
    new_hidden_state = np.tanh(W_hh * hidden_state + W_xh * input_val + b_h)
    
    # 2. Calculate output: y_t = W_hy * h_t + b_y
    # In many real RNNs, the output might go through another activation like Softmax, 
    # but for this raw demo, we keep it linear.
    output = W_hy * new_hidden_state + b_y
    
    return new_hidden_state, output

def main():
    print("--- Simple RNN Forward Pass (NumPy) ---\n")
    
    # --- 1. Define Parameters (Weights & Biases) ---
    # In a real training scenario, these would be learned. Here we set them manually.
    # We use scalar values for simplicity, but these are usually matrices/vectors.
    W_xh = 0.5   # Weight for input -> hidden
    W_hh = 0.8   # Weight for hidden -> hidden (Recurrent weight!!)
    W_hy = 1.0   # Weight for hidden -> output
    
    b_h = 0.0    # Bias for hidden
    b_y = 0.0    # Bias for output
    
    print(f"Parameters: W_xh={W_xh}, W_hh={W_hh}, W_hy={W_hy}\n")
    
    # --- 2. Define Input Sequence ---
    # Let's say this represents a sequence of 4 numbers: [1, 2, 3, 4]
    inputs = [1, 2, 3, 4]
    print(f"Input Sequence: {inputs}\n")
    
    # --- 3. Run the RNN Loop (Unrolling) ---
    # Initial hidden state is usually zeros
    prev_hidden_state = 0.0
    
    print("Beginning Time Steps:")
    for t, x_t in enumerate(inputs):
        # Perform one step
        new_hidden_state, output = simple_rnn_step(
            x_t, prev_hidden_state, W_hh, W_xh, W_hy, b_h, b_y
        )
        
        print(f"  Step {t+1}:")
        print(f"    Input (x_t): {x_t}")
        print(f"    Previous Hidden (h_t-1): {prev_hidden_state:.4f}")
        print(f"    New Hidden (h_t):        {new_hidden_state:.4f}")
        print(f"    Output (y_t):            {output:.4f}")
        print("-" * 30)
        
        # CRITICAL STEP: The new hidden state becomes the previous hidden state for the next step
        prev_hidden_state = new_hidden_state

    print("\nResult:")
    print("Notice how the Hidden State changes at each step.")
    print("It carries information from previous steps (weighted by W_hh) mixed with new input.")

if __name__ == "__main__":
    main()
