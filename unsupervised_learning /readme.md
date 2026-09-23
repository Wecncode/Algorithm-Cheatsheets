# Unsupervised Learning Cheatsheets

Welcome to the **Unsupervised Learning** section of the Algorithm Cheatsheets repository! 

Unsupervised Learning is a subcategory of Machine Learning where algorithms are trained on **unlabeled datasets**. Unlike Supervised Learning, there is no "correct answer" or target variable provided to the model. Instead, the algorithm's job is to act as a detective: exploring the data to discover hidden structures, patterns, and relationships on its own.

## Core Concepts

Unsupervised Learning algorithms generally fall into three main categories based on the type of structure they are trying to uncover:

1.  **Clustering:** Grouping unclassified data points into distinct clusters based on their similarities (e.g., Customer Segmentation).
2.  **Dimensionality Reduction:** Compressing data from a high-dimensional space into a lower-dimensional space while keeping the meaningful properties of the original data (e.g., Image Compression, Feature Extraction).
3.  **Association Rule Mining:** Discovering interesting relations and "if-then" rules between variables in large databases (e.g., Market Basket Analysis).

## Directory Contents

Here is what you will find in this directory. Each algorithm includes a theoretical Markdown cheatsheet and an accompanying Jupyter Notebook with Python code (`scikit-learn`, `scipy`, `mlxtend`, etc.).

### Clustering Algorithms
*   **K-Means Clustering:** [Cheatsheet](./k-means.md) 
    *   *Best for:* Fast, centroid-based grouping when you roughly know how many clusters you want.
*   **Hierarchical Clustering:** [Cheatsheet](./hierarchical.md) 
    *   *Best for:* Discovering nested structures and visualizing taxonomy via a dendrogram (without guessing $k$).
*   **DBSCAN:** [Cheatsheet](./density_based_spatial_clustering.md) 
    *   *Best for:* Finding arbitrary-shaped clusters and automatically detecting outliers/noise in dense spatial data.

### Dimensionality Reduction
*   **Principal Component Analysis (PCA):** [Cheatsheet](./principal-component-analysis.md) 
    *   *Best for:* Squashing highly correlated features into a few principal components to speed up training and visualize complex data.
*   **Singular Value Decomposition (SVD):** [Cheatsheet](./singular_value_decomposition.md) 
    *   *Best for:* Matrix factorization, handling sparse text data, and building the foundation of Recommendation Systems.

### Association Rules
*   **Apriori Algorithm:** [Cheatsheet](./apriori.md)
    *   *Best for:* Market Basket Analysis and finding out which items are frequently bought or consumed together.

## 🚀 How to use this section

*   **For Theory:** Read the `.md` files. They contain the mathematical intuition, pros and cons, evaluation metrics, and real-world use cases.
*   **For Practice:** Open the `code/` folder to see how to import, train, and evaluate these models using standard Python libraries.

---
*Back to the [Main Repository README](../README.md)*
