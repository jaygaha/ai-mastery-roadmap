# The Perceptron: The "Atom" of Neural Networks

Before we dive into massive neural networks with billions of connections, we need to understand the simplest possible unit: the **Perceptron**. 

Think of a perceptron as a tiny decision-making machine. It takes in some information, weighs it based on importance, and makes a simple "Yes" or "No" decision.

## How a Perceptron Makes Decisions

Imagine you're trying to decide whether to go for a run. You might consider two things:
1. Is the weather nice? ($x_1$)
2. Do I have enough time? ($x_2$)

A perceptron handles this by looking at three main parts:

- **Inputs ($x$):** The raw data. In our example, these are "Weather" and "Time."
- **Weights ($w$):** These represent **importance**. If you care a lot about the weather but only a little about time, "Weather" will have a high weight, and "Time" will have a low one.
- **Bias ($b$):** This is like your **inclination**. If you're naturally lazy, you might have a "negative bias" (you need extra convincing to go). If you’re a fitness fanatic, you might have a "positive bias" (you’re already halfway out the door).

### The Math (Made Simple)

The perceptron does a very simple calculation:
1. It multiplies each input by its weight (Input $\times$ Importance).
2. It adds them all together.
3. It adds the **Bias**.

```text
Decision Score = (Weather × Importance) + (Time × Importance) + Bias
```

Finally, it uses an **Activation Function** (like a light switch). If the score is positive, the switch flips to "1" (Go for a run!). If the score is zero or negative, it stays at "0" (Stay on the couch).


> This example shows how a perceptron handles numerical data to make a binary decision.

Imagine a bank uses a perceptron to decide if someone gets a loan. It looks at two things: **Credit Score** ($x_1$) and **Annual Income** ($x_2$).

- **Weights:** The bank decides Credit Score is twice as important as Income. 
  - $w_1 = 0.6$ (Credit)
  - $w_2 = 0.3$ (Income)
- **Bias:** The bank is strict. It sets a bias of **-0.5**, meaning you need a very high score to overcome this "starting penalty."

### The Calculation
An applicant has a good credit score (0.8) and decent income (0.7).

1. **Multiply & Sum:** $(0.8 \times 0.6) + (0.7 \times 0.3) = 0.48 + 0.21 = 0.69$
2. **Apply Bias:** $0.69 - 0.5 = 0.19$
3. **The Result:** The score is **0.19**. Since this is greater than 0, the "switch" flips to "1". **Loan Approved!**

What if they had a poor credit score (0.3) but high income (0.8)?
1. **Sum:** $(0.3 \times 0.6) + (0.8 \times 0.3) = 0.18 + 0.24 = 0.42$
2. **Bias:** $0.42 - 0.5 = -0.08$
3. **The Result:** The score is **-0.08**. Since it's negative, the switch stays at "0". **Loan Denied.**

## How the Perceptron Learns (The "Trial and Error" Rule)

A perceptron doesn't know the right weights and bias from the start. It has to learn them by making mistakes. This is the heart of **Machine Learning**.

The process is very human:
1. **Guess:** Start with random weights and a random bias.
2. **Predict:** Take a training example and guess the answer.
3. **Compare:** Look at the real answer.
4. **Correct:** If the guess was wrong, tweak the weights and bias slightly to do better next time.

### The Correction Formula

When the perceptron makes a mistake, it updates itself using this rule:

```text
New Weight = Old Weight + (Learning Rate × Error × Input)
New Bias = Old Bias + (Learning Rate × Error)
```

- **Learning Rate ($\alpha$):** Think of this as the "intensity" of the correction. If it’s high, the model makes big changes. If it’s low, it makes tiny, careful adjustments.
- **Error:** This is the difference between the **True Label** ($y$) and the **Predicted Label** ($\hat{y}$). 
  - If the model was correct, the Error is 0 (No change needed).
  - If the model guessed 0 but it should have been 1, the Error is positive (Increase the weights).
  - If the model guessed 1 but it should have been 0, the Error is negative (Decrease the weights).

> Below is a step-by-step walkthrough of how a perceptron learns an "AND" gate.

### Example: Learning to Click

Let's use a hypothetical scenario to demonstrate the perceptron learning rule. We want to classify whether a customer will click on an ad based on their age (x₁) and past website visits (x₂). We'll simplify the inputs to binary for clarity here.

| Age $(x_1)$ | Visits $(x_2)$ | Click $(y)$ |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |


This represents an **AND** logic gate: a click only happens if both age is "mature" (1) and visits are "frequent" (1).

Let's initialize:

- $w_1=0$
- $w_2=0$
- $b=0$
- Learning rate $\alpha=0.1$
- Threshold for activation = 0

**Iteration 1, Training Example 1: $(x_1=0, x_2=0, y=0)$**

1. Calculate weighted sum: $z=(0\ast0)+(0\ast0)+0=0$
2. Activation: $\hat{y}=step(0)=1$ (assuming $\geq0$ maps to 1)
3. Error: $(y-\hat{y})=(0-1)=-1$
4. Update weights: $w_1=0+0.1\ast(-1)\ast0=0$ $w_2=0+0.1\ast(-1)\ast0=0$ $b=0+0.1\ast(-1)=-0.1$ 

    Current parameters: $w_1=0,w_2=0,b=-0.1$

**Iteration 1, Training Example 2: $(x_1=0, x_2=1, y=0)$**

1. Calculate weighted sum: $z=(0\ast0)+(1\ast0)−0.1=−0.1$
2. Activation: $\hat{y}=step(−0.1)=0$
3. Error: $(y−\hat{y})=(0−0)=0$
4. No update needed. Current parameters: $w_1=0,w_2=0,b=−0.1$

**Iteration 1, Training Example 3: $(x_1=1, x_2=0, y=0)$**

1. Calculate weighted sum: $z=(1\ast0)+(0\ast0)−0.1=−0.1$
2. Activation: $\hat{y}=step(−0.1)=0$
3. Error: $(y−\hat{y})=(0−0)=0$
4. No update needed. Current parameters: $w_1=0,w_2=0,b=−0.1$

**Iteration 1, Training Example 4: $(x_1=1, x_2=1, y=1)$**

1. Calculate weighted sum: $z=(1\ast0)+(1\ast0)−0.1=−0.1$
2. Activation: $\hat{y}=step(−0.1)=0$
3. Error: $(y−\hat{y})=(1−0)=1$
4. Update weights: $w_1=0+0.1\ast(1)\ast1=0.1$ $w_2=0+0.1\ast(1)\ast1=0.1$ $b=−0.1+0.1\ast(1)=0$ 

    Current parameters: $w_1=0.1,w_2=0.1,b=0$

This process would continue over multiple epochs until the perceptron converges to weights and a bias that correctly classify all training examples (if the data is linearly separable). For the **AND** gate, it would eventually find weights like $w_1=0.1,w_2=0.1,b=−0.15$ (or scaled versions), where $0.1x_1+0.1x_2−0.15>0$ only when $x_1=1$ and $x_2=1.$

## The One Big Flaw: The XOR Problem

The perceptron is brilliant, but it has one major limitation: it can only think in straight lines (literally). 

If you can separate your data with a single straight line (called **Linear Separability**), the perceptron is your friend. But life isn't always that simple.

The most famous example is the **XOR (Exclusive OR) Gate**. 
- In an AND gate, you need both lights on to win. 
- In an OR gate, you need at least one light on. 
- In an **XOR** gate, you need **exactly one** light on.

If you plot these points on a graph, you'll find that there is **no way** to separate the "Yes" answers from the "No" answers with a single straight line.

> [!IMPORTANT]
> This limitation was so significant that it almost killed interest in neural networks in the late 1960s (a period called the **AI Winter**). It wasn't until scientists added "Hidden Layers" (creating Deep Neural Networks) that AI could solve complex problems like XOR.

---

## Real-World Impact

Even though simple perceptrons aren't used for modern Siri or ChatGPT, their logic is everywhere:

1. **Email Filtering:** A very simple "Spam" vs "Not Spam" decision. Does it have the word "FREE"? (+weight). Is it from a known contact? (-weight).
2. **Medical Triage:** Deciding if a patient is "High Risk" based on a few key factors like Age, Blood Pressure, and Symptoms.

---

## Exercises

Check your understanding by working through these manual and code-based exercises.

1. **Manual Perceptron Calculation:** A perceptron has two inputs, $x_1$ and $x_2$, with weights $w_1=0.5$ and $w_2=−0.2$. The bias is $b=0.1$. The activation function is a step function where output is 1 if the weighted sum $\geq0$, otherwise 0. Calculate the output ($\hat{y}$) for the following input sets: a. $x_1 = 1, x_2 = 0$ b. $x_1 = 0, x_2 = 1$ c. $x_1 = 1, x_2 = 1$
   - [Solution: _4_1_exercise.py](./_4_1_exercise.py)

2. **Perceptron Learning Rule:** Consider a perceptron with a single input $x$, an initial weight $w=0.3$, and an initial bias $b=−0.1$. The learning rate $\alpha=0.1$. The activation function is a step function (output 1 if weighted sum $\geq0$, else 0). Train the perceptron for one epoch using the following training examples:

    - $(x=1, y=1)$
    - $(x=0, y=0)$

   Show the updated weight and bias after processing each training example.

   - [Solution: _4_2_exercise.py](./_4_2_exercise.py)

3. **Linear Separability:** Consider the following dataset:

   | Feature 1 (x₁) | Feature 2 (x₂) | Class (y) |
   | --- | --- | --- |
   | 1 | 1 | 0 |
   | 1 | 2 | 0 |
   | 2 | 1 | 0 |
   | 2 | 2 | 1 |

   - Plot these points on a 2D graph. 
   - Can a single perceptron classify this data perfectly? 
   - Explain why or why not in terms of linear separability. 
   - If possible, sketch a line that could separate the classes.

   [Solution: _4_3_exercise.py](./_4_3_exercise.py)

## Conclusion

The Perceptron is where it all started. While it seems simple, it introduced the two concepts that make all modern AI work: **weights** (importance) and **learning** (trial and error).

In the next lesson, we’ll see how we can evolve from these simple "Yes/No" switches to something more flexible by looking at **Activation Functions**. We'll also see what happens when we start stacking these "atoms" together to build complex "molecules" – or what we today call **Deep Neural Networks**.