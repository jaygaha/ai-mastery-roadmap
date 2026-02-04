# Specialized Deep Learning Architectures and Applications

This module explores advanced neural network architectures designed for specific data types and tasks. While standard feedforward networks work well for tabular data, specialized architectures unlock the true potential of deep learning for images, sequences, and generative tasks.

## Module Contents

### 06.1 Convolutional Neural Networks (CNNs): Theory and Architecture
Located in: `_06_1_CNN_Theory_Architecture`

Master the fundamentals of image processing with deep learning:

- **The Convolution Operation**: Learn how filters detect patterns by sliding across images
- **Key Concepts**: Understand stride, padding, and how they affect output dimensions
- **Pooling Layers**: Reduce spatial dimensions while preserving important features
- **CNN Architecture**: See how conv layers, pooling, and dense layers work together
- **Hands-On Exercises**: Calculate output dimensions, implement max pooling, build a CNN

**What You'll Learn:**
- ✅ How CNNs "see" images by detecting hierarchical features
- ✅ The math behind convolution (filter * input = feature map)
- ✅ Why max pooling makes CNNs robust to small translations
- ✅ How to calculate the number of parameters in a CNN layer

---

### 06.2 Building Image Classification Models
Located in: `_06_2_Build_Image_CNN`

Apply CNN theory to build complete image classification pipelines:

- **Data Loading**: Using `image_dataset_from_directory` and preprocessing
- **Model Architecture**: Stacking Conv2D, Pooling, and Dense layers
- **Training & Evaluation**: Monitoring loss/accuracy and avoiding overfitting
- **Visualization**: Seeing what the model learns (filters) and how it improves
- **Augmentation**: Using data augmentation to improve generalization

**What You'll Learn:**
- ✅ How to structure a CNN project from scratch
- ✅ Using Keras preprocessing layers for data augmentation
- ✅ Visualizing learned filters and feature maps
- ✅ Comparing architecture variations (pooling, kernel size, depth)

---

### 06.3 RNN Theory & Architecture
Located in: `_06_3_RNN_Theory_Architecture`

Dive into sequential data processing:

- **Sequence Data Challenges**: Why standard networks fail with time-series/text.
- **The "Memory" State**: How RNNs maintain context over time.
- **Unrolling**: Visualizing the recurrent loop.
- **Hands-On**: A pure NumPy implementation to see the math in action.
- **Vanishing Gradients**: Understanding the limitations that led to LSTMs.

**What You'll Learn:**
- ✅ The difference between One-to-Many, Many-to-One, and Seq2Seq.
- ✅ How $h_t = \tanh(W h_{t-1} + U x_t)$ works.
- ✅ Why context matters for words like "bank" (river vs. money).

---

### 06.4 Building Simple RNNs
Located in: `_06_4_Build_Simple_RNN`

Put RNN theory into practice with hands-on implementations:

- **Data Preparation**: Transform time series and text into RNN-ready formats
- **Time Series Forecasting**: Predict future values using sliding window sequences
- **Text Classification**: Build a sentiment analyzer with Embedding + SimpleRNN
- **Practical Tips**: When to use SimpleRNN vs. LSTM/GRU

**What You'll Learn:**
- ✅ How to create sliding window sequences for time series
- ✅ The tokenization → padding → embedding pipeline for text
- ✅ Building and training SimpleRNN models in Keras
- ✅ Visualizing predictions vs. actual values

---

### 06.5 Transfer Learning & Pre-trained Models
Located in: `_06_5_Transfer_Learning_Pretrained`

Leverage the power of state-of-the-art models without needing massive datasets:

- **Concept**: Why "re-inventing the wheel" is inefficient for AI.
- **Strategies**: Feature Extraction vs. Fine-Tuning.
- **Architectures**: Using ResNet, VGG, MobileNet, etc.
- **Hands-On**: Adapting a model trained on ImageNet to a custom task.

**What You'll Learn:**
- ✅ How to load Keras Applications with pre-trained weights.
- ✅ Freezing and unfreezing layers strategies.
- ✅ Fine-tuning models with low learning rates.
- ✅ Solving complex problems with small datasets.

---

### 06.6 Explore Advanced DL: Unstructured Data & Multimodal Models
Located in: `_06_6_Explore_Advanced_DL`

Go beyond numbers and categories by integrating **textual data** (customer feedback) into your churn predictions:

- **Text Vectorization**: Understand TF-IDF and Bag-of-Words.
- **Word Embeddings**: Learn how vectors capture semantic meaning.
- **Multimodal Architectures**: Build networks that see (data) AND read (text).
- **Advanced Exercises**: Tune vocabulary size, compare LSTM vs. Pooling, and optimize hyperparameters.

**What You'll Learn:**
- ✅ How to preprocess raw text for Deep Learning.
- ✅ Building a model with multiple inputs (Functional API).
- ✅ The trade-off between simple text features (TF-IDF) and sequence models (LSTM).

---


*More specialized architectures coming soon:*
- *LSTM and GRU Networks*
- *Attention Mechanisms and Transformers*
- *Generative Adversarial Networks (GANs)*