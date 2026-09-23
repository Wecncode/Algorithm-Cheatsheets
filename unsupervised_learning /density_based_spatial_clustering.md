# DBSCAN Cheatsheet
*(Density-Based Spatial Clustering of Applications with Noise)*

**Type:** Unsupervised Learning (Clustering)  
**Output:** Cluster labels for each data point (Points labeled as `-1` are considered noise/outliers)

## Intuition
Imagine looking at a satellite map of the world at night. The brightly lit, densely packed cities are your **clusters**. The dark, rural areas separating the cities represent the boundaries. A random lonely light in the middle of an ocean? That's **noise**.

Unlike K-Means (which assumes clusters are spherical and requires you to guess how many clusters exist), **DBSCAN** simply groups points that are closely packed together and marks points that lie alone in low-density regions as outliers. 

## The Core Concepts

DBSCAN classifies every data point into one of three categories based on two main hyperparameters:

1.  **Core Point:** A point that has at least `min_samples` within its $\epsilon$ (epsilon) radius. It is deep inside a cluster.
2.  **Border Point:** A point that does *not* have enough neighbors to be a Core Point, but it falls within the $\epsilon$ radius of a Core Point. It forms the "edge" of a cluster.
3.  **Noise Point (Outlier):** A point that is neither a Core Point nor a Border Point. It is entirely isolated.

## Key Hyperparameters
*   **`eps` ($\epsilon$):** The maximum distance between two points for them to be considered neighbors. 
    *   *Too small:* Most data will be labeled as noise (-1).
    *   *Too large:* Multiple distinct clusters will merge into one giant cluster.
    *   *Pro-Tip:* Use a k-distance graph (Nearest Neighbors) to find the optimal "elbow" value for `eps`.
*   **`min_samples`:** The minimum number of points required to form a dense region (a cluster).
    *   Rule of thumb: `min_samples` $\ge$ Dimensions + 1 (For 2D data, use at least 3, but 4 or 5 is standard).

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **No $k$ Required:** You do not need to specify the number of clusters in advance. | **Varying Densities:** Struggles if clusters have vastly different densities (a single `eps` value won't work for all). |
| **Arbitrary Shapes:** Can find non-linear, weirdly shaped clusters (like spirals or moons) that K-Means fails at. | **Curse of Dimensionality:** Distance metrics break down in high-dimensional space, making `eps` hard to tune. |
| **Built-in Outlier Detection:** Naturally isolates noise points instead of forcing them into a cluster. | **Sensitive to Hyperparameters:** Getting `eps` and `min_samples` exactly right can be tricky. |

## Evaluation Metrics
Since clustering is unsupervised, evaluation is often subjective, but you can use:
*   **Silhouette Score:** Measures how similar an object is to its own cluster compared to other clusters. *(Note: Favors convex/spherical clusters, so it might artificially score DBSCAN poorly on complex shapes).*
*   **Visual Inspection:** For 2D/3D data, plotting the clusters is the most reliable method.

## Real-World Projects & Use Cases
1.  **Anomaly/Fraud Detection:** Identifying fraudulent credit card transactions or network intrusions that fall outside dense behavioral clusters.
2.  **Geospatial Analysis:** Finding hubs of activity from GPS coordinate data (e.g., identifying popular pickup spots for Uber/Lyft).
3.  **Image Processing:** Segmenting pixels into objects and background.
4.  **Recommendation Systems:** Grouping similar users where distinct dense sub-communities exist.

## Sample Python Implementation (scikit-learn)

```python
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# 2. Load Data (Using make_moons for a non-linear example)
X, _ = make_moons(n_samples=500, noise=0.1, random_state=42)

# 3. SCALE THE FEATURES (Crucial for distance-based algorithms!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Initialize and train the model
# tuning eps and min_samples is the most important step
dbscan = DBSCAN(eps=0.2, min_samples=5)
cluster_labels = dbscan.fit_predict(X_scaled)

# 5. Analyze the output
# Labels = -1 are noise/outliers
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")

# 6. Bonus: Visualize the clusters
plt.figure(figsize=(8, 5))
# Plot points, coloring by cluster label
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis', s=50, edgecolors='k')
plt.title(f"DBSCAN Clustering (Clusters: {n_clusters}, Noise: {n_noise})")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.show()
