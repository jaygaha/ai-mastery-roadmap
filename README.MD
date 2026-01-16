# AI Course Roadmap: From Fundamentals to Intermediate Applications

Welcome to the AI Course Roadmap! This project is designed to guide you through the exciting world of Artificial Intelligence, starting from the basics of Python and AI fundamentals to more complex and intermediate applications.

## Project Overview

This repository serves as a structured learning path. Each module focuses on specific topics, providing code examples, exercises, and resources to help you master AI concepts using Python.

## Getting Started

### Prerequisites

- **Python 3.x**: Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).
- **IDE**: A code editor like VS Code or PyCharm is recommended.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ai_course_roadmap.git
    cd ai_course_roadmap
    ```

2.  **Explore the modules:**
    Navigate to the specific module directory you are interested in.

## Course Structure

The course is divided into modules. As we progress, new modules will be added here.

### 1. AI Foundation and Python Essentials
Located in: `_1_AI_Foundation_and_Python_Essentials`

This module covers the groundwork necessary for AI development:
- Python syntax and data structures.
- Basic programming concepts.
- Introduction to AI terminology.

#### Crash Course
Located in: `_1_AI_Foundation_and_Python_Essentials/crash-course`

A quick dive into Python essentials including:
- **Data Structures & Algorithms**: `_1_dsa`
- **Control Flow**: `_2_ctrlflw`

#### Practical Python for Data Manipulation: NumPy Fundamentals
Located in: `_1_AI_Foundation_and_Python_Essentials/data-manipulation-NumPy-fundamentals`

Master the core technology for numerical computing in Python.

#### Data Analysis with Pandas
Located in: `_1_AI_Foundation_and_Python_Essentials/pandas`

Learn to manipulate and analyze structured data using Series and DataFrames.

#### Case Study: Customer Churn Prediction
Located in: `_1_AI_Foundation_and_Python_Essentials/case_study_customer_churn_prediction`

Apply your skills to a real-world business problem: predicting customer churn.

### 2. Data Exploration and Preprocessing
Located in: `_2_Data_Exploration_and_Preprocessing`

Before feeding data into models, you must understand and prepare it. This module covers:

#### Understanding Data Types, Sources, and Loading Techniques
Located in: `_2_Data_Exploration_and_Preprocessing/_01_Understanding_Data_Types_Sources_and_Loading_Techniques`

Master the basics of data:
- Types: Numerical, Categorical, Text.
- Sources: Databases, Files, APIs.
- Loading: Using Pandas to ingest data.

#### Exploratory Data Analysis (EDA) with Pandas and Matplotlib
Located in: `_2_Data_Exploration_and_Preprocessing/_02_EDA_with_Pandas_and_Matplotlib`

Learn to:
- Inspect data structure and types.
- Visualize distributions and relationships.
- Identify patterns and anomalies.

#### Data Cleaning: Handling Missing Values, Outliers, and Inconsistencies
Located in: `_2_Data_Exploration_and_Preprocessing/_03_Data_Cleaning`

Ensure data quality by:
- Identifying and handling missing values.
- Detecting and treating outliers.
- Correcting data inconsistencies.

#### Feature Engineering and Selection for Model Performance
Located in: `_2_Data_Exploration_and_Preprocessing/_04_model_performance`

Unlock the true potential of your data:
- **Feature Engineering**: Creating new signals from existing data.
- **Feature Selection**: Choosing the most impactful features for your models.

#### Data Scaling, Normalization, and Encoding
Located in: `_2_Data_Exploration_and_Preprocessing/_05_Data_Scaling_Normalization_and_Encoding`

Prepare your data for efficient learning:
- **Scaling & Normalization**: Adjusting numerical data to a common scale.
- **Encoding**: Converting categorical variables into numerical format.

#### Preparing Customer Churn Case Study Data for Modeling
Located in: `_2_Data_Exploration_and_Preprocessing/_06_Preparing_Customer_Churn_Case_Study_Data_for_Modeling`

Prepare customer churn case study data for machine learning models:
- **Feature Engineering**: Creating new signals from existing data.
- **Feature Selection**: Choosing the most impactful features for your models.
- **Data Scaling, Normalization, and Encoding**: Adjusting numerical data to a common scale.
 - **Encoding**: Converting categorical variables into numerical format.

### 3. Core Machine Learning Algorithms
Located in: `_3_Core_Machine_Learning_Algorithms`

Dive into the algorithms that power AI:

#### Supervised Learning Paradigms: Regression vs Classification
Located in: `_3_Core_Machine_Learning_Algorithms/_01_Supervised_Learning_Paradigms_Regression_vs_Classification`

- Understand the fundamental difference between predicting numbers (Regression) and predicting categories (Classification).

#### Linear Regression: Theory, Assumptions, and Practical Implementation
Located in: `_3_Core_Machine_Learning_Algorithms/_02_Linear_Regression`

- Learn the principles of linear modeling and OLS.
- Master the 5 key assumptions for reliable predictions.
- Apply it to synthetic and real-world data.

#### Logistic Regression: Theory, Probabilistic Classification, and Practical Implementation
Located in: `_3_Core_Machine_Learning_Algorithms/_03_Logistic_Regression`

- Understand probabilistic classification and the Sigmoid function.
- Experiment with decision thresholds for precision/recall trade-offs.
- Interpret model coefficients for feature importance.

#### Decision Trees and Ensemble Methods (Random Forests): Theory and Implementation
Located in: `_3_Core_Machine_Learning_Algorithms/_04_Decision_Trees_and_Ensemble_Methods`

- Learn how Decision Trees split data (Entropy, Gini).
- Understand Ensemble Learning and Random Forests.
- Tune hyperparameters for optimal performance.

#### Model Evaluation Metrics for Regression: MAE, MSE, R2
Located in: `_3_Core_Machine_Learning_Algorithms/_05_Model_Evaluation_Metrics_for_Regression`

- **Evaluate Performance**: Don't just predict, measure.
- **Key Metrics**: Master MAE (Average Miss), MSE (Strict), and R2 (Accuracy).
- **Interpretation**: Learn to translate "R2=0.8" into business terms.

#### Model Evaluation Metrics for Classification: Accuracy, Precision, Recall, F1-Score
Located in: `_3_Core_Machine_Learning_Algorithms/_06_Model_Evaluation_Metrics_for_Classification`

- **Beyond Accuracy**: Why "99% accuracy" can be misleading.
- **The Confusion Matrix**: Understanding False Positives vs False Negatives.
- **Precision vs Recall**: Making strategic business decisions based on error types.

#### Applying Classic ML Algorithms to the Churn Prediction Case Study
Located in: `_3_Core_Machine_Learning_Algorithms/_07_Applying_Classic_ML_Algorithms`

- **Real-World Application**: Apply Logistic Regression, Decision Trees, and Random Forests to customer churn prediction.
- **Model Comparison**: Compare performance using evaluation metrics and make data-driven decisions.
- **Hyperparameter Tuning**: Optimize model performance by adjusting key parameters.
- **Threshold Adjustment**: Learn to balance precision and recall based on business priorities.
- **Feature Importance**: Discover which customer characteristics drive churn predictions.
- **Business Recommendations**: Make informed model selection decisions for production deployment.
+

### 4. Introduction to Deep Learning with TensorFlow
Located in: `_4_Introduction_to_Deep_Learning_with_TensorFlow`

Explore the next frontier of AI:

#### From Traditional ML to Deep Learning: The Concept of Neural Networks
Located in: `_4_Introduction_to_Deep_Learning_with_TensorFlow/_04_1_The_Concept_of_Neural_Networks`

- Learn how neural networks discover their own features.
- Understand how complex patterns are built from simple edges and shapes.

#### The Perceptron Model: The "Atom" of Neural Networks
Located in: `_4_Introduction_to_Deep_Learning_with_TensorFlow/_04_2_Perception_Model_Explained`

- Master the basic components: Inputs, Weights, and Bias.
- Understand the "Learning Rule" that allows machines to improve.
- Learn about the XOR problem and why we need multi-layer networks.

#### Activations & Loss Functions: The Engine of Learning
Located in: `_4_Introduction_to_Deep_Learning_with_TensorFlow/_04_3_Activations_Loss_Functions`

- **Activation Functions:** The "Spark" (ReLU, Sigmoid, Softmax).
- **Loss Functions:** The "Scorecard" (MSE, Cross-Entropy).
- **Goal:** How networks decide what to learn and how to measure mistakes.

#### Gradient Descent & Backpropagation: How Networks Learn
Located in: `_4_Introduction_to_Deep_Learning_with_TensorFlow/_04_4_Gradient_Backpropagation_Explained`

- **The Goal:** Minimizing the loss function.
- **Gradient Descent:** Adjusted weights to move "downhill" towards the answer.
- **Backpropagation:** The "Blame Game" – calculating who is responsible for the error.

### Setup TensorFlow Keras
Located in: `_4_Introduction_to_Deep_Learning_with_TensorFlow/_04_5_Setup_TensorFlow_Keras`

- **Installation:** Setting up your deep learning environment (Windows, Mac, Linux).
- **Core Concepts:** Understanding the relationship between TensorFlow (Backend) and Keras (Frontend).
- **First Model:** Building and training a "Hello World" neural network.

#### Building and Training a Simple Feedforward Neural Network
Located in: `_4_Introduction_to_Deep_Learning_with_TensorFlow/_04_6_Build_Train_NeuralNet`

- **Build:** Creating a Sequential model with input, hidden, and output layers.
- **Train:** Using epochs and batch size to learn patterns from real data.
- **Evaluate:** Checking performance on unseen test data.

### 5. Developing and Optimizing Neural Networks
Located in: `_5_Optimize_Neural_Networks`

This module explores the art of fine-tuning neural networks to improve their performance.

#### Multi-Layer Perceptrons (MLPs) for Complex Classification and Regression
Located in: `_5_Optimize_Neural_Networks/_05_1_MLPs_for_Complex_Classification_and_Regression`

- **Architecture:** Understanding Input, Hidden, and Output layers.
- **Activation Functions:** Using ReLU, Sigmoid, and Softmax.
- **Implementation:** Building MLPs with Keras for multi-class and regression tasks.

#### Training Neural Networks: Epochs, Batch Size, and Optimizers (Adam, SGD)
Located in: `_5_Optimize_Neural_Networks/_05_2_Train_Network_Optimizers`

Master the critical parameters that control neural network training:
- **Epochs:** Learn to balance underfitting and overfitting by controlling training iterations.
- **Batch Size:** Understand the trade-offs between memory usage, training speed, and convergence stability.
- **Optimizers:** Compare SGD and Adam, understanding when each is most effective.
- **Hands-On Practice:** Three comprehensive exercises to build intuition for these parameters.
- **Real-World Impact:** See how these choices affect fraud detection, product recommendations, and customer churn prediction.

## How to Contribute

Contributions are welcome! If you have suggestions for improvements, new topics, or bug fixes, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
