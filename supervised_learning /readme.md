# Supervised Learning Cheatsheets

Welcome to the **Supervised Learning** section of the Algorithm Cheatsheets repository! 

Supervised Learning is a subcategory of Machine Learning where algorithms are trained using **labeled datasets**. The model learns to map an input to an output based on historical examples. Once trained, the model can predict the output for new, unseen data.

## Core Concepts

Supervised Learning algorithms are generally split into two main tasks based on the type of output variable they predict:

1.  **Classification:** Predicting a discrete class label (e.g., "Spam" vs. "Not Spam", or "Cat" vs. "Dog").
2.  **Regression:** Predicting a continuous numerical value (e.g., House Prices, Temperature, or Stock Prices).

## Directory Contents

Here is what you will find in this directory. Each algorithm has a theoretical Markdown cheatsheet and an accompanying Jupyter Notebook with Python code (`scikit-learn`, `xgboost`, etc.).

### Regression Algorithms
*   **Linear Regression:** [Cheatsheet](./linear_regression.md) 
    *   *Best for:* Simple, linear relationships and establishing a baseline.
*   **Polynomial Regression:** [Cheatsheet](./polynomial_regression.md) 
    *   *Best for:* Non-linear, curved relationships using linear math.

### Classification (and Regression) Algorithms
*   **Logistic Regression:** [Cheatsheet](./logistic_regression.md) 
    *   *Best for:* Binary classification and outputting probabilities.
*   **K-Nearest Neighbors (KNN):** [Cheatsheet](./knn.md) 
    *   *Best for:* Simple clustering tasks where distance equals similarity.
*   **Support Vector Machines (SVM):** [Cheatsheet](./svm.md) 
    *   *Best for:* Complex, high-dimensional data with clear margins of separation.
*   **Naive Bayes:** [Cheatsheet](./naive_bayes.md) 
    *   *Best for:* Text classification, spam filtering, and NLP baselines.
*   **Decision Trees:** [Cheatsheet](./trees.md) 
    *   *Best for:* Highly interpretable, rule-based decision making.

### Ensemble Methods
*   **Random Forest:** [Cheatsheet](./random_forest.md) 
    *   *Best for:* High accuracy out-of-the-box, robust to overfitting, handling tabular data.
*   **Boosting (AdaBoost, XGBoost):** [Cheatsheet](./boosting.md) 
    *   *Best for:* Winning Kaggle competitions, squeezing maximum performance from structured data.

## How to use this section

*   **For Theory:** Read the `.md` files. They contain the mathematical intuition, pros and cons, and real-world use cases.
*   **For Practice:** Open the `code/` folder to see how to import, train, predict, and evaluate these models using standard Python libraries.

---
*Back to the [Main Repository README](../README.md)*
