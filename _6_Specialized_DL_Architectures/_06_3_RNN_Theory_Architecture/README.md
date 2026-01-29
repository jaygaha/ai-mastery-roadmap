# Recurrent Neural Networks (RNNs): Theory and Architecture for Sequence Data

Recurrent Neural Networks (RNNs) are a special type of neural network designed to handle **sequential data**—data where the order of information matters. Timestamps, lists of words, and stock prices are all examples where "what came before" is critical for understanding "what comes next."

Unlike traditional feedforward networks that treat every input as an isolated event, RNNs have a "memory." They remember what they've seen previously and combine that context with the current input to make a decision.

## The Challenge of Sequence Data

Imagine trying to read a sentence one word at a time, but immediately forgetting the previous word as soon as you see the new one.
*   Input: "The" -> Brain: "Okay."
*   Input: "cat" -> Brain: "Okay." (Forgot "The")
*   Input: "sat" -> Brain: "Okay." (Forgot "cat")

By the time you reach "sat," you have no idea *who* sat. This is how standard Feedforward Neural Networks (like MLPs) operate; they lack internal memory state involving time.

**Why Standard Networks Struggle:**
1.  **No Context**: Predicting the next word in "The cat sat on the..." requires knowing "The cat sat." A standard network only sees "the" (the current input) and misses the "mat."
2.  **Fixed Input Size**: Standard networks require a fixed input size (e.g., 10 features). But sentences can be 5 words or 50 words long.

## Recurrent Neural Network (RNN) Architecture

RNNs solve this by keeping a **Hidden State** ($h_t$)—a vector of numbers that represents the network's current "thought" or "context."

When an RNN processes a sequence, it does so step-by-step:
1.  It takes the **Current Input** ($x_t$).
2.  It blends it with the **Previous Hidden State** ($h_{t-1}$).
3.  It produces a **New Hidden State** ($h_t$) and an **Output** ($y_t$).

### Visualizing "Unrolling"

We often visualize RNNs as a loop that passes information to itself. To understand it better, we "unroll" the loop over time:

```text
    [Input x_1]      [Input x_2]      [Input x_3]
        |                |                |
        v                v                v
  [State h_0] -->  [State h_1] -->  [State h_2] --> ...
        |                |                |
        v                v                v
    [Output y_1]     [Output y_2]     [Output y_3]
```

*   **Time Step 1**: The network takes $x_1$ and an initial empty state $h_0$. It calculates $h_1$.
*   **Time Step 2**: It takes $x_2$ AND the state $h_1$ (which "remembers" $x_1$). It calculates $h_2$.
*   **Time Step 3**: It takes $x_3$ AND the state $h_2$ (which now "remembers" both $x_1$ and $x_2$).

This chain allows the network to carry information from the very beginning of the sentence to the very end.

### The Math (simplified)

The core equation for calculating the new memory state is:

$$ h_t = \tanh(W \cdot h_{t-1} + U \cdot x_t + b) $$

*   $h_t$: New hidden state.
*   $h_{t-1}$: Previous hidden state.
*   $x_t$: Current input.
*   $W, U$: Weight matrices (parameters) that the network learns.
*   $\tanh$: Activation function that squashes values between -1 and 1, keeping the numbers stable.

> [!NOTE]
> Even though we process a sequence of 100 items, we reuse the **same weights** ($W$ and $U$) at every single step. This is key: the RNN learns one set of rules for "how to update memory" and applies it universally.

## Hands-On: NumPy Implementation

To truly understand the architecture, it helps to see the math in code without any "magic" libraries like Keras.

**File:** [`_3_1_simple_rnn_numpy.py`](./_3_1_simple_rnn_numpy.py)

This script manually implements the forward pass of a basic RNN. It demonstrates:
1.  Defining random input data (sequences).
2.  Initializing weights.
3.  Looping through time steps to update the hidden state.

## Types of RNN Architectures

Depending on the task, RNNs can be wired differently:

1.  **One-to-Many**: One input -> Sequence of outputs.
    *   *Example:* Image Captioning (Image -> "A dog playing in the grass").
2.  **Many-to-One**: Sequence of inputs -> One output.
    *   *Example:* Sentiment Analysis ("This movie was great..." -> Positive).
3.  **Many-to-Many (Synced)**: Sequence -> Sequence (same length).
    *   *Example:* Video Frame Labeling (Frame 1 -> Label 1, Frame 2 -> Label 2).
4.  **Many-to-Many (Asynchronous)**: Sequence -> Sequence (different length). Also called **Seq2Seq**.
    *   *Example:* Machine Translation (English sentence -> French sentence).

## The "Vanishing Gradient" Problem

While impactful, simple RNNs have a major flaw. As you go deeper into a sequence (backpropagating through time), the gradients (signals used to update weights) often get smaller and smaller until they effectively vanish.

**Analogy:** Imagine a game of "Telephone" (or Chinese Whispers). By the time the message reaches the 10th person, the original meaning is often lost. Similarly, a simple RNN at step 100 struggles to "hear" the error signal from step 1, making it hard to learn long-term connections.

**The Solution:** This limitation led to **LSTMs (Long Short-Term Memory)** and **GRUs (Gated Recurrent Units)**, which utilize special "gates" to protect the memory flow. We will cover these in the next module.

## Exercises

Test your understanding with these conceptual scenarios.
*Solutions are available in [SOLUTIONS.md](./SOLUTIONS.md).*

1.  **Architecture Match**: Which RNN type fits best?
    *   Translating a voice recording (sequence of audio waves) to a text transcript.
    *   Classifying a tweet as "Spam" or "Not Spam".
    *   Generating a music melody from a single starting note.

2.  **Memory Tracing**: Convert the sentence "I love deep learning" into steps.
    *   At step 2 (input "love"), what information does the hidden state $h_1$ likely hold?
    *   Why is knowing "I" important for interpreting "love" (contextually)?

3.  **Gradient Issues**:
    *   Why might a simple RNN fail to predict the last word of a textbook based on the first word?