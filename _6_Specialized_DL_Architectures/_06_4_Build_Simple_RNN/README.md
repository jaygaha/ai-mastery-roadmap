# Building a Simple RNN for Time Series Forecasting or Text Data

Have you ever tried to predict what comes next in a conversation without knowing what was said before? It's nearly impossible! That's exactly the challenge neural networks face with sequential data—and it's why Recurrent Neural Networks (RNNs) were invented.

**Think of an RNN like reading a book:** you don't start each sentence by forgetting everything that came before. Instead, you carry forward context from previous sentences. RNNs do the same thing—they maintain a "memory" of what they've seen, allowing them to understand sequences like text, audio, or time series data.

## Why RNNs Matter

Before diving into the technical details, let's understand when you'd actually use an RNN:

- **Predicting stock prices**: The price tomorrow depends on patterns from previous days
- **Sentiment analysis**: "I love this movie... NOT!" requires understanding the whole sentence
- **Language translation**: Word order matters—"dog bites man" ≠ "man bites dog"
- **Speech recognition**: Each sound is connected to what came before

## Understanding the Simple RNN Architecture

A simple RNN layer processes sequences one step at a time. At each step, it looks at two things:
1. The **current input** (like today's stock price)
2. The **hidden state** from the previous step (its "memory" of the past)

### The Core Idea: A Memory That Updates

Imagine you're watching a movie and keeping mental notes. At each scene:
- You take in what's happening now (current input)
- You combine it with what you remember (previous hidden state)
- You update your mental notes (new hidden state)

Mathematically, this looks like:

```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
```

**Breaking this down in plain English:**
- `h_t` = your updated memory (hidden state at time t)
- `h_{t-1}` = what you remembered from before
- `x_t` = what's happening right now (current input)
- `W_hh`, `W_xh` = learned weights (how much to trust memory vs. new info)
- `tanh` = squashes values between -1 and 1 (keeps things stable)

### Unrolling the RNN: Seeing the Big Picture

To visualize how an RNN processes a sequence, imagine "unrolling" it through time:

```
Input:    x_1 ──→ x_2 ──→ x_3 ──→ ... ──→ x_T
           ↓       ↓       ↓               ↓
RNN:     [RNN] ─→ [RNN] ─→ [RNN] ─→ ... ─→ [RNN]
           ↓       ↓       ↓               ↓
Hidden:   h_1 ──→ h_2 ──→ h_3 ──→ ... ──→ h_T
```

The key insight: **the same RNN cell (with the same weights) is applied at each step**. This weight sharing is what allows RNNs to handle sequences of varying lengths.

### Input and Output Shapes: The Practical Details

When implementing RNNs in Keras, understanding shapes is crucial:

**Input Shape: `(batch_size, timesteps, features)`**
- `batch_size`: How many sequences you process at once (e.g., 32 sentences)
- `timesteps`: Length of each sequence (e.g., 10 words per sentence)
- `features`: Dimensions per timestep (e.g., 128-dimensional word embedding)

**Output Shape depends on `return_sequences`:**
- `return_sequences=False` → `(batch_size, units)`: Returns only the **final** hidden state. Use for tasks like sentiment classification where you need one answer per sequence.
- `return_sequences=True` → `(batch_size, timesteps, units)`: Returns hidden state at **every** timestep. Use when stacking RNN layers or for sequence-to-sequence tasks.

## Data Preparation for RNNs

Good data preparation is half the battle. Let's look at two common scenarios.

### Time Series Data Preparation

For predicting future values, we create "sliding windows" of past observations:

**Example:** Given temperatures `[20, 21, 22, 23, 24, 25]` with `look_back=3`:
- Input: `[20, 21, 22]` → Output: `23`
- Input: `[21, 22, 23]` → Output: `24`
- Input: `[22, 23, 24]` → Output: `25`

Each input sequence becomes a "sample" that the RNN learns from.

> [!NOTE]
> **Code Example:** [Simple RNN Data Preparation](./_4_1_rnn_time_series_data_preparation.py)

### Text Data Preparation

Text requires a few extra steps to convert words into numbers:

1. **Tokenization**: Split "I love cats" → `["I", "love", "cats"]`
2. **Numericalization**: Map words to IDs → `[1, 2, 3]`
3. **Padding**: Make all sequences the same length → `[1, 2, 3, 0, 0]`

The padding ensures RNNs can process batches efficiently (all sequences in a batch must have the same length).

> [!NOTE]
> **Code Example:** [Simple RNN Text Data Preparation](./_4_2_rnn_text_data_preparation.py)

When combined with an `Embedding` layer, padded sequences `(num_samples, max_length)` become rich feature representations `(num_samples, max_length, embedding_dim)`—exactly what the RNN expects!

## Building a Simple RNN Model with Keras

Keras makes building RNNs straightforward with the `SimpleRNN` layer.

### Key SimpleRNN Parameters

| Parameter | Description | When to Adjust |
|-----------|-------------|----------------|
| `units` | Size of hidden state (memory capacity) | Increase for complex patterns |
| `activation` | Activation function (default: `tanh`) | Usually keep default |
| `return_sequences` | Return all states or just the last | `True` for stacked RNNs |
| `return_state` | Also return the final state separately | For encoder-decoder models |

### Implementing a Simple RNN for Time Series Forecasting

Here's the complete workflow: generate data → scale → create sequences → train → predict.

> [!NOTE]
> **Complete Example:** [Simple RNN Time Series Forecasting](./_4_3_rnn_time_series_forecasting.py)

**Key points:**
- We scale data to [0, 1] using `MinMaxScaler` (RNNs train better with normalized inputs)
- After prediction, we "inverse transform" to get back to original units
- The model learns to predict the next value given a window of past values

### Implementing a Simple RNN for Text Classification

For sentiment analysis, we add an `Embedding` layer before the RNN:

```
Text → Tokenize → Pad → Embedding → SimpleRNN → Dense(sigmoid)
```

The embedding layer converts word IDs into dense vectors, capturing semantic meaning (similar words get similar vectors).

> [!NOTE]
> **Complete Example:** [Simple RNN Text Classification](./_4_4_rnn_text_classification.py)

## Practical Considerations and Limitations

Simple RNNs have a significant weakness: **they struggle with long-term dependencies**.

### The Vanishing Gradient Problem

When training, gradients must flow backward through time. With long sequences, these gradients get multiplied repeatedly and often shrink to nearly zero—the network "forgets" what happened early in the sequence.

**Rule of thumb:** Simple RNNs work well for sequences under ~20-30 timesteps. For longer sequences, use **LSTMs** or **GRUs**, which have special "gates" to preserve important information.

> [!TIP]
> If your simple RNN isn't learning well on longer sequences, don't keep tuning hyperparameters—switch to LSTM or GRU instead!

## Exercises

### Exercise 1: Time Series with Increased Look-Back

Modify the time series forecasting example to use `look_back=20` instead of 10. Train the model and observe:
- How does the training loss compare?
- Plot predicted vs. actual values for the test set

**Hint:** After training, use `model_ts.predict(X_test)` and `scaler.inverse_transform()` to get predictions in the original scale.

> [!NOTE]
> **Solution:** [Exercise 1 - Time Series with look_back=20](./_4_5_1_exercise.py)

---

### Exercise 2: Text Classification with More Data

Expand the text classification example:
1. Add 2-3 more positive and negative sentences
2. Increase `embedding_dim` to 32 and `SimpleRNN` units to 64
3. How does training accuracy change with these modifications?

**Hint:** Make sure your `labels` array matches the new number of sentences!

> [!NOTE]
> **Solution:** [Exercise 2 - Expanded Text Classification](./_4_5_2_exercise.py)

## Next Steps

You've now built your first RNNs for both time series and text data! However, simple RNNs are just the beginning. In upcoming lessons, we'll explore:

- **LSTMs (Long Short-Term Memory)**: Solve the vanishing gradient problem with memory cells and gates
- **GRUs (Gated Recurrent Units)**: A simpler alternative to LSTMs with similar performance
- **Transfer Learning**: Leverage pre-trained models to accelerate development

These advanced architectures unlock the full power of sequential modeling, enabling you to tackle complex tasks like machine translation, speech synthesis, and more.