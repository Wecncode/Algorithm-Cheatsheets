# Hierarchical Clustering Cheatsheet

**Type:** Unsupervised Learning (Clustering)  
**Output:** A tree-like hierarchy of data points (Dendrogram) and cluster labels

## Intuition
Unlike K-Means, which requires you to guess the number of clusters ($k$) upfront, **Hierarchical Clustering** groups data over a variety of scales by creating a cluster tree. 

There are two main approaches:
1.  **Agglomerative (Bottom-Up - Most Common):** Starts with every single data point as its own cluster. It then finds the two closest clusters and merges them. It repeats this process until all points are merged into one giant root cluster.
2.  **Divisive (Top-Down):** Starts with all data points in one giant cluster and recursively splits them until every point is on its own.

## The Dendrogram
The ultimate output of Hierarchical Clustering is a **Dendrogram**. 
*   The x-axis represents the individual data points.
*   The y-axis represents the **distance** (or dissimilarity) between clusters.
*   Horizontal lines represent merges. The *height* of the horizontal line tells you exactly how far apart the two clusters were when they merged.
*   **To get final clusters:** You draw a horizontal line across the dendrogram at a specific height threshold. The number of vertical lines you intersect is your number of clusters!

## Linkage Criteria (How to measure distance between *clusters*)
To merge clusters, you need to know the distance between them. But how do you measure the distance between a cluster of 5 points and a cluster of 10 points?

1.  **Ward's Method (Default/Best):** Minimizes the total variance within the merged clusters. Very similar to how K-Means optimizes, usually creates the most evenly sized clusters.
2.  **Single Linkage (Min):** The distance between two clusters is the distance between their *closest* points. (Prone to "chaining" where clusters get stretched out).
3.  **Complete Linkage (Max):** The distance between two clusters is the distance between their *farthest* points. (Tends to create compact, spherical clusters).
4.  **Average Linkage:** The average distance between all points in cluster A and all points in cluster B.

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **No $k$ Required Upfront:** You can look at the dendrogram *after* training to decide the best number of clusters. | **Computationally Expensive:** Big O complexity is $O(n^3)$ or $O(n^2 \log n)$. Terrible for massive datasets. |
| **Highly Interpretable:** The dendrogram provides a beautiful, intuitive visual of the data's underlying structure. | **Irreversible:** Once a merge is done, it cannot be undone. It cannot re-assign points later like K-Means can. |
| **Captures Hierarchy:** Excellent for data that naturally has a taxonomy (e.g., biology, document grouping). | **Sensitive to Outliers:** Especially when using Single or Complete linkage. |

## Evaluation Metrics
*   **Visualizing the Dendrogram:** Look for the longest vertical lines that don't have any horizontal lines crossing them; that's usually the best place to cut.
*   **Silhouette Score:** Measures intra-cluster tightness vs. inter-cluster separation.
*   **Cophenetic Correlation Coefficient:** Measures how faithfully the dendrogram preserves the pairwise distances of the original unclustered data.

## Real-World Projects & Use Cases
1.  **Biology & Genetics:** Creating phylogenetic trees to map evolutionary relationships between different species based on DNA similarity.
2.  **Customer Segmentation:** Grouping customers into broad categories, and then sub-categories (e.g., High Spenders $\rightarrow$ High Spending Millennials vs. Boomers).
3.  **Document Clustering:** Organizing a massive library of text into hierarchical topics (e.g., Science $\rightarrow$ Physics $\rightarrow$ Quantum Mechanics).
4.  **Stock Market Analysis:** Grouping stocks that exhibit similar price movements over time to build diversified portfolios.

## Sample Python Implementation (scipy & scikit-learn)

*Usually, we use `scipy` to draw the dendrogram, and `scikit-learn` to actually get the cluster labels.*

```python
# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# 2. Load Data (Example)
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.5, random_state=42)

# 3. SCALE THE FEATURES (Crucial for distance algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Draw the Dendrogram (using SciPy)
plt.figure(figsize=(10, 6))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance (Ward)')
# Create linkage matrix and plot
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.axhline(y=10, color='r', linestyle='--') # Example cut line
plt.show()

# 5. Get Cluster Labels (using Scikit-Learn)
# Based on the dendrogram, let's say we want 4 clusters
hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
cluster_labels = hc.fit_predict(X_scaled)

# 6. Evaluate or Use Labels
print(f"First 10 cluster assignments: {cluster_labels[:10]}")
