# Principal Component Analysis (PCA) Cheatsheet

**Type:** Unsupervised Learning (Dimensionality Reduction)  
**Output:** A transformed dataset with fewer features (components) than the original, ranked by importance.

## Intuition
Imagine you have a 3D object, like a teapot. If you shine a flashlight on it, it casts a 2D shadow on the wall. If you hold the flashlight at a bad angle, the shadow just looks like a blob. But if you hold it at the *perfect* angle, the 2D shadow captures almost exactly what the 3D teapot looks like.

**PCA finds that perfect angle.** It takes a dataset with hundreds of features (dimensions) and compresses it down to just a few new features (Principal Components) while retaining as much of the original information (**variance**) as possible.

## The Core Concepts (Linear Algebra)

PCA doesn't just drop existing features; it mathematically constructs *new* ones.

1.  **Covariance Matrix:** PCA first looks at how every single feature in the dataset relates to every other feature to find redundancies (e.g., if "Square Footage" and "Number of Rooms" are highly correlated, they can be combined).
2.  **Eigenvectors:** These are the new "directions" (axes) that PCA calculates. The 1st Principal Component (PC1) is the direction where the data varies the most. PC2 is the second most varying direction, strictly perpendicular (orthogonal) to PC1.
3.  **Eigenvalues:** These are numbers attached to the Eigenvectors that tell you *how much* variance (information) that specific component holds. 

## Choosing the Number of Components
How do you know how many dimensions to compress your data down to? 
You look at the **Explained Variance Ratio**. 
*   PC1 might explain 45% of the data's variance.
*   PC2 might explain 25%.
*   PC3 might explain 10%.
*   *Rule of Thumb:* Keep adding components until your cumulative explained variance reaches a threshold you are happy with (usually **80% to 95%**).

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Fights the Curse of Dimensionality:** Drastically reduces the number of features, making other ML models train much faster. | **Loss of Interpretability:** The new "Principal Components" are mathematical amalgams. PC1 doesn't mean "Age" or "Salary" anymore; it's a mix of everything. |
| **Noise Reduction:** By dropping the least important components, PCA naturally filters out random noise in the data. | **Linear Assumption:** PCA assumes relationships between features are linear. It fails on complex, folded non-linear data (use t-SNE or UMAP instead). |
| **Data Visualization:** Allows you to visualize 50-dimensional data by squishing it down to a 2D or 3D scatter plot. | **Must Scale Data:** If you don't standardize your data, variables with large scales will completely dominate the components. |

## Evaluation Metrics
*   **Cumulative Explained Variance:** The primary metric. (e.g., "Our first 4 components explain 92% of the total variance").

## Real-World Projects & Use Cases
1.  **Image Compression:** Reducing high-resolution images down to a fraction of their size while maintaining the core visual structure (e.g., JPEG compression concepts).
2.  **Genomics:** Visualizing DNA microarray data, compressing thousands of gene expressions into a 2D plot to find clusters of similar patients.
3.  **Algorithmic Trading:** Compressing the daily price movements of 500 stocks in the S&P 500 into just 3 or 4 overarching "market trends" (Principal Components).
4.  **Preprocessing Pipeline:** Running PCA *before* training a slow algorithm like SVM or a Neural Network to speed up training times by 10x.

## Sample Python Implementation (scikit-learn)

```python
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# 2. Load Data (Example: 30 features)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# 3. SCALE THE FEATURES (ABSOLUTELY MANDATORY FOR PCA!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Initialize and fit PCA
# Let's say we want to keep 95% of the variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 5. Analyze the results
print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of features: {X_pca.shape[1]}")
print(f"Explained Variance Ratio per component: {pca.explained_variance_ratio_}")
print(f"Total Variance Explained: {np.sum(pca.explained_variance_ratio_):.4f}")

# 6. Bonus: Plot the Cumulative Explained Variance
pca_full = PCA().fit(X_scaled)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot (Explained Variance)')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.grid(True)
plt.show()
