# Singular Value Decomposition (SVD) Cheatsheet

**Type:** Unsupervised Learning (Dimensionality Reduction / Matrix Factorization)  
**Output:** Three factorized matrices ($U$, $\Sigma$, $V^T$) that represent the original data's latent structures.

## Intuition
Imagine you have a giant spreadsheet of Users rating Movies. Most of the spreadsheet is empty (because nobody watches *every* movie). **SVD** is a mathematical algorithm that deconstructs this giant, sparse matrix into three smaller, dense matrices. 

By multiplying these three smaller matrices back together, you can approximate the original giant matrix—and magically, it will fill in the missing blanks, essentially predicting what rating a user *would* give to a movie they haven't seen yet!

In broader terms, SVD breaks complex data down into its foundational "latent" (hidden) concepts. 

## The Math (Matrix Factorization)
The fundamental theorem of SVD states that *any* matrix $A$ can be factored into three distinct matrices:

$$A = U \cdot \Sigma \cdot V^T$$

1.  **$A$ (Input Data):** The original matrix (e.g., $m$ users $\times$ $n$ movies).
2.  **$U$ (Left Singular Vectors):** Represents the relationship between the rows of $A$ and the hidden latent concepts (e.g., How much does User 1 like the "Action" concept?). Size: $m \times r$.
3.  **$\Sigma$ (Sigma - Singular Values):** A diagonal matrix containing numbers sorted in descending order. These numbers represent the "importance" or "weight" of each hidden concept. Size: $r \times r$.
4.  **$V^T$ (Right Singular Vectors):** Represents the relationship between the columns of $A$ and the hidden latent concepts (e.g., How much does this Movie belong to the "Action" concept?). Size: $r \times n$.

## Truncated SVD (Dimensionality Reduction)
Standard SVD is mathematically exact, but in Machine Learning, we usually only care about the *most important* concepts. 

In **Truncated SVD**, we look at the $\Sigma$ matrix and simply throw away all the small singular values at the bottom (which usually just represent noise). By keeping only the top $k$ values, we drastically compress the data while retaining the core patterns.

*Note: PCA actually uses SVD under the hood! The difference is that PCA centers the data (subtracts the mean) first, while standard/Truncated SVD does not.*

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Handles Sparse Data:** Truncated SVD is fantastic for sparse matrices (like text data or user ratings) where PCA would require dense memory. | **Computationally Expensive:** Calculating the exact SVD for massive matrices takes a lot of time and memory ($O(n^3)$). |
| **Foundation of Recommenders:** The absolute gold standard for Collaborative Filtering recommendation engines. | **Hard to Interpret:** The "latent concepts" it discovers are purely mathematical. Concept 1 might be a bizarre mix of "Comedy + 80s Movies - Tom Hanks". |
| **Exact Mathematical Solution:** Unlike neural networks, SVD provides a deterministic, mathematically optimal decomposition. | **Static:** If you add a new user or movie to your dataset, you often have to recalculate the entire SVD from scratch. |

## Evaluation Metrics
*   **Explained Variance:** Just like PCA, you can measure how much of the original matrix's information is retained after truncating to $k$ components.
*   **RMSE (Root Mean Squared Error):** In recommendation systems, how close the reconstructed matrix's predicted ratings are to the actual known ratings.

## Real-World Projects & Use Cases
1.  **Recommender Systems:** The famous algorithm that won the $1 Million Netflix Prize. Predicting user ratings for unseen items (Collaborative Filtering).
2.  **Latent Semantic Analysis (LSA):** In Natural Language Processing (NLP), SVD is applied to a Document-Term matrix to group words by their hidden context, identifying that "car" and "automobile" mean the same thing.
3.  **Image Compression:** Storing an image using only the top 50 singular values instead of thousands of individual pixels, saving massive amounts of disk space.
4.  **Noise Reduction:** Removing static/noise from signals or images by discarding the smallest singular values (which usually capture the high-frequency noise).

## Python Implementation (scikit-learn)

*While `numpy.linalg.svd` provides the exact math, `sklearn`'s `TruncatedSVD` is highly optimized for machine learning and sparse data.*

```python
# 1. Import libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# 2. Load Data (Example: NLP / Text Data)
documents = [
    "Machine learning is fascinating.",
    "Artificial intelligence and machine learning.",
    "The stock market is crashing.",
    "Finance and stock market trends."
]

# 3. Create a sparse Document-Term Matrix (TF-IDF)
vectorizer = TfidfVectorizer()
X_sparse = vectorizer.fit_transform(documents)

# 4. Initialize and fit Truncated SVD (LSA)
# Let's compress the vocabulary down to just 2 latent concepts/topics
n_components = 2
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X_sparse)

# 5. Analyze the Results
print(f"Original shape: {X_sparse.shape} (4 documents, {X_sparse.shape[1]} words)")
print(f"Reduced shape: {X_reduced.shape} (4 documents, 2 latent topics)")

print(f"\nExplained Variance Ratio: {svd.explained_variance_ratio_}")
print(f"Total Variance Explained: {np.sum(svd.explained_variance_ratio_):.4f}")

# 6. Interpretation (Looking at the 'V^T' matrix)
vocab = vectorizer.get_feature_names_out()
for i, comp in enumerate(svd.components_):
    # Get top 3 words for this latent topic
    top_words_idx = comp.argsort()[-3:][::-1]
    top_words = [vocab[idx] for idx in top_words_idx]
    print(f"Topic {i+1} Top Words: {top_words}")
