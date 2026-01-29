# Solutions: RNN Theory & Architecture

## 1. Architecture Match

*   **Translating a voice recording (audio wave sequence) to a text transcript.**
    *   **Architecture:** Many-to-Many (Asynchronous / Seq2Seq)
    *   **Why:** The input (audio frames) and output (words) are both sequences, but their lengths are rarely the same (e.g., 5 seconds of audio != 5 words).

*   **Classifying a tweet as "Spam" or "Not Spam".**
    *   **Architecture:** Many-to-One
    *   **Why:** The input is a sequence of words (variable length), but the output is a single label/category.

*   **Generating a music melody from a single starting note.**
    *   **Architecture:** One-to-Many
    *   **Why:** You provide one input (the seed note), and the network auto-regressively generates a long sequence of subsequent notes.

## 2. Memory Tracing

**Sentence:** "I love deep learning"

*   **At step 2 (input "love"):**
    *   Hidden state $h_1$ (calculated from step 1) holds the representation of the word **"I"**.
    *   When the network sees "love" ($x_2$), it combines it with $h_1$ ("I").
    *   The resulting $h_2$ conceptually represents the phrase **"I love"**.

*   **Why is knowing "I" important?**
    *   In English, "love" can be a verb ("I love...") or a noun ("The love...").
    *   Seeing "I" previously tells the network that "love" is likely acting as a **verb** in this context. This helps disambiguate meaning.

## 3. Gradient Issues

*   **Why might it fail for long sequences?**
    *   This is the **Vanishing Gradient Problem**.
    *   During training (backpropagation), the error signal is multiplied by the weight matrix $W$ repeatedly for every time step.
    *   If the weights are small (e.g., < 1), multiplying them 100 times (e.g., $0.5^{100}$) results in an incredibly tiny number (effectively zero).
    *   This means the network at step 1 gets **zero feedback** about how it affected the output at step 100. It effectively "stops learning" connections that span long distances.
