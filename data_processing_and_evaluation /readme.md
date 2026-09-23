# Data Processing & Evaluation Cheatsheets

Welcome to the **Data Processing & Evaluation** section of the Algorithm Cheatsheets repository! 

In Machine Learning, algorithms are only as good as the data you feed them and the metrics you use to judge them. The phrase *"Garbage In, Garbage Out"* is the golden rule of data science. This section covers the crucial steps required *before* you train a model (Preprocessing) and the techniques used *after* training to ensure it actually works (Evaluation & Tuning).

## Core Concepts

This directory is split into two primary phases of the Machine Learning lifecycle:

1.  **Data Preprocessing (The "Before"):** Cleaning messy real-world data, handling missing values, scaling numbers so algorithms can digest them fairly, and translating text categories into mathematical representations.
2.  **Model Evaluation & Selection (The "After"):** Rigorously testing models on unseen data using Cross-Validation, tuning their hyperparameters for peak performance, and selecting the correct mathematical metric to judge success.

## Directory Contents

Here is what you will find in this directory. Each topic includes a theoretical Markdown cheatsheet and an accompanying Jupyter Notebook with Python code (using `pandas` and `scikit-learn`).

### Data Preprocessing & Feature Engineering
*   **Handling Missing Data:** [Cheatsheet](./handling_missing_data.md) 
    *   *Covers:* MCAR, MAR, MNAR, Deletion vs. Imputation, Simple vs. KNN Imputers.
*   **Feature Scaling:** [Cheatsheet](./feature_scaling.md) 
    *   *Covers:* Normalization (Min-Max) vs. Standardization (Z-Score), and preventing data leakage.
*   **Categorical Encoding:** [Cheatsheet](./encoding.md)
    *   *Covers:* Ordinal Encoding, One-Hot Encoding (Dummy Variables), and Target/Mean Encoding.

### Evaluation & Optimization
*   **Evaluation Metrics:** [Cheatsheet](./evaluation_metrics.md) 
    *   *Covers (Classification):* Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC.
    *   *Covers (Regression):* MAE, MSE, RMSE, R-Squared.
*   **Cross-Validation & Hyperparameter Tuning:** [Cheatsheet](./cross_validation_and_hyperparameter_tuning.md) 
    *   *Covers:* K-Fold Cross Validation, Stratified splits, Grid Search, and Random Search.

## How to use this section

*   **For Theory:** Read the `.md` files. They contain the intuition, mathematical formulas, pros and cons, and rules of thumb for when to use which technique.
*   **For Practice:** Open the `code/` folder to see how to build robust, leakage-free pipelines using `scikit-learn` that automatically handle scaling, encoding, and imputation inside cross-validation loops.

---
*Back to the [Main Repository README](../README.md)*
