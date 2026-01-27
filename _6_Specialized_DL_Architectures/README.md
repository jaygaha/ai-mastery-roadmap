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

*More specialized architectures coming soon:*
- *RNNs and LSTMs for sequence data*
- *Attention mechanisms and Transformers*
- *Generative Adversarial Networks (GANs)*