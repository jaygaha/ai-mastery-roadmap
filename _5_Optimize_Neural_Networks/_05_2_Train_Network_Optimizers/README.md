# Training Neural Networks: Epochs, Batch Size, and Optimizers (Adam, SGD)

You've built your neural network architecture—now comes the crucial part: teaching it to learn. Training a neural network is like teaching someone a new skill through practice. You need to decide how many times they'll practice (epochs), how much material to cover in each session (batch size), and what learning strategy works best (optimizer).

In this module, we'll explore the three key parameters that control how your neural network learns from data. These aren't just technical settings—they're the difference between a model that learns effectively and one that struggles or even fails to improve.

## The Training Process: Epochs and Batch Size

Think of training as a repetitive learning process. Just like studying for an exam, you can't absorb everything in one go—you need to break it down into manageable chunks and review multiple times.

### Epochs

An **epoch** is one complete pass through your entire training dataset. During an epoch, the neural network sees every single training example once and updates its weights accordingly.

Here's a simple way to think about it: if you have 1,000 images of cats and dogs, training for 10 epochs means your model will study each of those 1,000 images exactly 10 times. With each pass, the model gets a little better at recognizing patterns.

**The Goldilocks Problem:**
- **Too few epochs** → Your model is **underfitted**. It's like cramming the night before an exam—you haven't learned enough to perform well.
- **Too many epochs** → Your model is **overfitted**. It's like memorizing the exact practice problems instead of understanding the concepts. Your model performs great on training data but fails on new, unseen data.
- **Just right** → Your model learns the underlying patterns without memorizing noise.

**Real-World Example:** Imagine training a model to classify cat and dog images with 10,000 photos. One epoch means the network has seen all 10,000 images once. Training for 20 epochs means it reviews the entire collection 20 times, refining its understanding of what makes a cat a cat and a dog a dog with each pass.


### Batch Size

**Batch size** determines how many training examples the model processes before updating its weights. Think of it as deciding how many practice problems to work through before checking your answers and adjusting your approach.

Instead of processing the entire dataset at once (which would be like trying to memorize an entire textbook in one sitting), we divide the data into smaller batches. The model looks at one batch, calculates how wrong it was, and adjusts accordingly.

**Small Batch Size (e.g., 16 or 32):**
- ✅ **More frequent updates** → The model adjusts its weights more often, which can help it learn faster
- ✅ **Less memory needed** → Great for training on limited hardware
- ✅ **Better at escaping bad solutions** → The "noisy" updates can help the model avoid getting stuck
- ❌ **Noisier training** → The loss might jump around more, making training less stable
- ❌ **Slower per epoch** → More updates mean more computational overhead

**Large Batch Size (e.g., 256 or 512):**
- ✅ **Smoother training** → Updates are based on more data, so they're more stable
- ✅ **Faster processing** → Modern GPUs can process large batches very efficiently
- ❌ **More memory required** → You need powerful hardware
- ❌ **Risk of getting stuck** → Might settle into suboptimal solutions

**Practical Example:** Let's say you're training a customer churn prediction model with 100,000 customer records.

- With `batch_size=32`: The model processes 32 customers at a time, updates weights, then moves to the next 32. One epoch = 100,000 ÷ 32 = **3,125 updates**
- With `batch_size=1024`: The model processes 1,024 customers at a time. One epoch = 100,000 ÷ 1,024 ≈ **98 updates**

Notice how batch size dramatically affects how many times your model updates per epoch!

## Optimizers: Your Model's Learning Strategy

If epochs and batch size determine *how* and *when* your model learns, **optimizers** determine *how well* it learns. An optimizer is the algorithm that adjusts your neural network's weights to minimize errors.

Think of it like this: You're trying to find the lowest point in a hilly landscape while blindfolded. The optimizer is your strategy for getting there—do you take big steps or small ones? Do you remember which direction worked before? Do you adjust your step size as you go?

### Gradient Descent: The Foundation

Gradient Descent is the basic optimization strategy. It works by:
1. Calculating how wrong the model is (the loss)
2. Figuring out which direction to adjust the weights (the gradient)
3. Taking a step in that direction
4. Repeating until the model stops improving

The **learning rate** controls how big each step is. Too big, and you might overshoot the best solution. Too small, and training takes forever.

**Variants of Gradient Descent:**

- **Stochastic Gradient Descent (SGD):**
    - Updates weights after *every single* training example
    - ✅ Fast updates, can escape bad solutions
    - ❌ Very noisy, erratic training path

- **Mini-Batch Gradient Descent:**
    - Updates weights after processing a *batch* of examples (most common approach)
    - ✅ Balances speed and stability
    - ✅ Works efficiently on modern GPUs
    - ❌ Requires tuning the learning rate and batch size

### Adaptive Learning Rate Optimizers

Beyond basic SGD, more advanced optimizers adapt the learning rate during training, often for each parameter individually. This can lead to faster convergence and better performance.

#### Adam (Adaptive Moment Estimation): The Smart Optimizer

Adam is the most popular optimizer in deep learning, and for good reason—it's like having an intelligent autopilot for your training process. While basic SGD uses the same learning rate for all weights, Adam is smarter: it adapts the learning rate for each individual parameter.

**Why Adam is Special:**

1. **Adaptive Learning Rates:** Different weights get different learning rates based on how they've been behaving. If a weight has been consistently moving in one direction, Adam gives it a bigger learning rate. If it's been jumping around, Adam slows it down.

2. **Momentum (Memory of the Past):** Adam remembers which direction has been working and keeps moving that way, like a ball rolling downhill. This helps it converge faster and avoid getting stuck.

3. **Smart Scaling:** Adam automatically adjusts how much each parameter should change based on the history of gradients. Parameters that rarely get updated get larger updates when they do; frequently updated parameters get smaller, more careful adjustments.

**How It Works (Simplified):**
- Adam keeps track of two things: the average direction gradients have been pointing (momentum) and how much they've been varying (variance)
- It uses these to give each weight a customized learning rate
- Early in training, it corrects for the fact that these averages start at zero

**Why Use Adam?**
- ✅ Works well "out of the box" with minimal tuning
- ✅ Handles different types of data and problems effectively
- ✅ Computationally efficient
- ✅ Generally faster convergence than basic SGD

**Real-World Example:** When training a language model, some words appear frequently ("the", "and") while others are rare ("antidisestablishmentarianism"). Adam automatically adjusts how much to learn from each word based on how often it appears, ensuring the model learns effectively from both common and rare patterns.

## Practical Examples and Demonstrations

We'll use a simplified neural network training scenario with Keras (TensorFlow) to illustrate these concepts, focusing on the configuration of epochs, batch size, and optimizers. We will continue to refer to our churn prediction case study where the goal is to classify customers as likely to churn (1) or not churn (0).

First, let's set up some dummy data for demonstration, representing preprocessed features and churn labels.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dummy data for demonstration, similar to churn prediction features
np.random.seed(42)
num_samples = 10000
num_features = 10

# Create synthetic features (e.g., monthly charges, contract length, etc.)
X = np.random.rand(num_samples, num_features) * 100 # Scale for better feature distribution

# Create synthetic target variable (churn or not churn)
# Make it somewhat dependent on features to simulate a learnable pattern
y = (X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.rand(num_samples) * 10 > 70).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (essential for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a simple neural network model
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid') # Output layer for binary classification
    ])
    return model

print(f"Training data shape: {X_train_scaled.shape}")
print(f"Testing data shape: {X_test_scaled.shape}")
```

### Experimenting with Epochs and Batch Size

We'll train the same model with different epoch and batch size configurations to observe their impact.

```python
# Model 1: Moderate epochs, standard batch size
print("\n--- Training Model 1: Epochs=10, Batch Size=32 ---")
model_1 = create_model()
model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_1 = model_1.fit(X_train_scaled, y_train,
                        epochs=10, batch_size=32,
                        validation_split=0.1, verbose=1)

# Model 2: More epochs, smaller batch size
print("\n--- Training Model 2: Epochs=20, Batch Size=16 ---")
model_2 = create_model()
model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_2 = model_2.fit(X_train_scaled, y_train,
                        epochs=20, batch_size=16,
                        validation_split=0.1, verbose=1)

# Model 3: Fewer epochs, larger batch size
print("\n--- Training Model 3: Epochs=5, Batch Size=256 ---")
model_3 = create_model()
model_3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_3 = model_3.fit(X_train_scaled, y_train,
                        epochs=5, batch_size=256,
                        validation_split=0.1, verbose=1)

# Evaluate models on test set
print("\n--- Evaluation ---")
loss_1, acc_1 = model_1.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model 1 (Epochs=10, Batch=32) Test Accuracy: {acc_1:.4f}")

loss_2, acc_2 = model_2.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model 2 (Epochs=20, Batch=16) Test Accuracy: {acc_2:.4f}")

loss_3, acc_3 = model_3.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model 3 (Epochs=5, Batch=256) Test Accuracy: {acc_3:.4f}")
```

**What to Observe:** When you run this code, pay attention to the "steps per epoch" in the training output. This number changes based on batch size:

- `batch_size=32`: Each epoch has `8000 / 32 = 250` steps (8000 is 80% of 10,000 samples)
- `batch_size=16`: Each epoch has `8000 / 16 = 500` steps
- `batch_size=256`: Each epoch has `8000 / 256 ≈ 31` steps

**Expected Results:**
- **Model 2** (20 epochs, batch=16): Likely the best performance due to more training time and frequent updates, but watch for overfitting
- **Model 1** (10 epochs, batch=32): Balanced approach, good baseline performance
- **Model 3** (5 epochs, batch=256): Fastest training but may not converge to the best solution

This demonstrates that there's no "perfect" configuration—it's always a trade-off between training time, memory usage, and final performance.

### Experimenting with Optimizers (Adam vs. SGD)

Now, let's compare the performance of different optimizers. We'll fix the epochs and batch size to isolate the optimizer's effect.

```python
# Model 4: Using Adam optimizer
print("\n--- Training Model 4: Optimizer='adam', Epochs=10, Batch Size=32 ---")
model_4 = create_model()
model_4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_4 = model_4.fit(X_train_scaled, y_train,
                        epochs=10, batch_size=32,
                        validation_split=0.1, verbose=1)

# Model 5: Using SGD optimizer with a learning rate
print("\n--- Training Model 5: Optimizer='sgd', Epochs=10, Batch Size=32 ---")
model_5 = create_model()
# It's crucial to set a learning rate for SGD, as it's often more sensitive
sgd_optimizer = keras.optimizers.SGD(learning_rate=0.01)
model_5.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history_5 = model_5.fit(X_train_scaled, y_train,
                        epochs=10, batch_size=32,
                        validation_split=0.1, verbose=1)

# Evaluate models on test set
print("\n--- Optimizer Evaluation ---")
loss_4, acc_4 = model_4.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model 4 (Adam) Test Accuracy: {acc_4:.4f}")

loss_5, acc_5 = model_5.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Model 5 (SGD) Test Accuracy: {acc_5:.4f}")
```


**What You'll See:** Model 4 (Adam) will almost certainly outperform Model 5 (SGD) in both speed and final accuracy. Why?

- **Adam** adapts its learning rate automatically, so it "just works" even without perfect tuning
- **SGD** with a fixed learning rate is very sensitive—if the learning rate is too high, training becomes unstable; too low, and it barely learns

This is why Adam has become the default choice for most deep learning practitioners. SGD can match or even beat Adam's performance, but only with careful tuning of the learning rate and often the addition of momentum.

## Exercises

These exercises will help you develop an intuition for how epochs, batch size, and optimizers affect training. Run each experiment and observe the results carefully.

### Exercise 1: Understanding Underfitting vs. Overfitting

**Goal:** See firsthand what happens when you train for too few or too many epochs.

**Tasks:**
1. Train `create_model()` for **1 epoch** with `batch_size=64` using the Adam optimizer
2. Evaluate and record both validation accuracy (from training) and test accuracy
3. Train a fresh model for **100 epochs** with `batch_size=64`
4. Evaluate and record both validation accuracy and test accuracy

**Questions to Answer:**
- Which model shows signs of **underfitting** (poor performance on both training and test data)?
- Which model might be **overfitting** (great training performance but worse test performance)?
- What's the sweet spot? How many epochs would you choose?

**Hint:** Look at the difference between validation accuracy and test accuracy. A large gap suggests overfitting.

---

### Exercise 2: The Impact of Batch Size

**Goal:** Understand how batch size affects training dynamics and convergence.

**Tasks:**
1. Train two models with `optimizer='adam'` and `epochs=15`
2. **Model A:** `batch_size=4` (very small)
3. **Model B:** `batch_size=512` (very large)
4. Monitor the training output carefully

**Questions to Answer:**
- Which model has more steps per epoch? Why?
- Which model's loss curve looks more "jumpy" or erratic?
- Which model trains faster (in terms of wall-clock time per epoch)?
- Which achieves better final test accuracy?

**Hint:** Small batches = more updates per epoch but noisier. Large batches = fewer, smoother updates.

---

### Exercise 3: SGD with Momentum vs. Adam

**Goal:** See how adding momentum to SGD improves performance and compare it to Adam.

**Tasks:**
1. Train a model with **SGD + Momentum**:
   ```python
   sgd_momentum = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
   ```
2. Use `epochs=10` and `batch_size=32`
3. Compare this model's performance to:
   - The original SGD model (without momentum) from the examples
   - The Adam model from the examples

**Questions to Answer:**
- How does momentum improve SGD's performance?
- Does SGD with momentum match Adam's performance?
- Why might you still prefer Adam for most projects?

**Hint:** Momentum helps SGD converge faster by "remembering" which direction was working before.

---

## Real-World Applications

Understanding epochs, batch size, and optimizers isn't just academic—these choices directly impact real-world AI systems. Let's see how:

### Financial Fraud Detection

**The Challenge:** Banks need to catch fraudulent transactions in real-time from millions of daily transactions. The data is highly imbalanced—fraud is rare (maybe 0.1% of transactions).

**Training Strategy:**
- **Batch Size:** Small to moderate (32-64) ensures the model sees fraud examples frequently enough to learn from them
- **Epochs:** Carefully tuned (10-20) to learn fraud patterns without overfitting to normal transactions
- **Optimizer:** Adam excels here because fraud patterns are sparse and varied. It adapts quickly to rare but important signals

**Why It Matters:** Missing fraud costs money; false alarms annoy customers. The right training configuration helps find the balance.

---

### E-commerce Product Recommendations

**The Challenge:** Amazon, Netflix, and similar platforms need to recommend products/content from millions of options based on billions of user interactions.

**Training Strategy:**
- **Batch Size:** Large (256-512) to process massive datasets efficiently and leverage powerful GPU clusters
- **Epochs:** Moderate (5-15) because the model is retrained frequently with fresh user data
- **Optimizer:** Adam or AdaGrad handle the diverse user preferences well—some products are popular (dense gradients), others niche (sparse gradients)

**Why It Matters:** Better recommendations = more sales and happier users. Fast training means the system can adapt to trends quickly.

---

### Customer Churn Prediction (Our Case Study)

**The Challenge:** Identify which customers are likely to cancel their subscription so you can intervene with retention offers.

**Training Strategy:**
- **Epochs:** Too few (< 5) and the model won't learn the subtle patterns of churn behavior. Too many (> 50) and it might memorize individual customers instead of learning general patterns.
- **Batch Size:** Medium (32-128) balances learning from diverse customer segments while maintaining stable training.
- **Optimizer:** Adam is ideal because churn signals vary widely—some customers churn due to price (common signal), others due to poor support experiences (rare but important).

**Why It Matters:** Each prevented churn saves the company money. A well-trained model with the right parameters can identify at-risk customers early enough to take action.

---

**Key Takeaway:** There's no universal "best" configuration. The optimal epochs, batch size, and optimizer depend on your data, problem, and resources. Experimentation and validation are essential!

## Summary and Next Steps

Congratulations! You now understand the three critical training parameters that can make or break your neural network:

- **Epochs:** How many times your model reviews the entire dataset (balance between underfitting and overfitting)
- **Batch Size:** How many examples to process before updating weights (trade-off between speed, memory, and stability)
- **Optimizers:** The learning strategy (Adam for most cases, SGD when you need fine control)

These aren't just knobs to turn randomly—they're strategic choices that depend on your data, problem, and computational resources.

### What's Next?

Even with perfect epochs, batch size, and optimizer settings, neural networks face a persistent challenge: **overfitting**. In the next module, we'll explore **regularization techniques**—powerful strategies like dropout, L1/L2 regularization, and early stopping that help your model generalize better to unseen data.

Think of regularization as teaching your model to focus on the signal and ignore the noise. Combined with what you've learned here, you'll be able to build neural networks that not only train well but also perform reliably in production.

**Ready to level up?** Complete the exercises above to solidify your understanding, then move on to regularization!