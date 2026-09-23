# K-Means Clustering Cheatsheet

**Type:** Unsupervised Learning (Clustering)  
**Output:** A cluster label (from $0$ to $k-1$) for each data point and the coordinates of the cluster centers.

## Intuition
K-Means is a centroid-based clustering algorithm. Its goal is to partition a dataset into $k$ distinct, non-overlapping subgroups (clusters). 

It tries to make the intra-cluster data points as similar as possible while keeping the clusters as different as possible. It does this by finding the center point (**centroid**) of each cluster and assigning each data point to the centroid it is closest to.

## How it Works (The Algorithm)
K-Means uses an iterative process known as **Expectation-Maximization**:

1.  **Initialization:** Randomly select $k$ points in the feature space to be the initial centroids. (Modern implementations use `k-means++` to spread these initial points out smartly).
2.  **Assignment (Expectation):** Calculate the distance (usually Euclidean) between every data point and every centroid. Assign each data point to its closest centroid.
3.  **Update (Maximization):** Calculate the mean (average) of all the data points assigned to each cluster. Move the centroid to this new mean location.
4.  **Repeat:** Repeat steps 2 and 3 until the centroids stop moving (convergence) or a maximum number of iterations is reached.

## Choosing 'k' (The Elbow Method)
You MUST tell the algorithm how many clusters ($k$) to look for. Since you usually don't know this in advance, you use the **Elbow Method**.

1.  Run K-Means for a range of $k$ values (e.g., 1 to 10).
2.  For each $k$, calculate the **WCSS (Within-Cluster Sum of Squares)**, also known as **Inertia**. This is the sum of the squared distances between each point and its centroid.
3.  Plot WCSS against $k$. As $k$ increases, WCSS will always drop. 
4.  Look for the "elbow" — the point where the drop in WCSS drastically slows down. That is your optimal $k$.

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Fast and Efficient:** Scales incredibly well to massive datasets (Time complexity is linear: $O(n)$). | **Requires $k$:** You must specify the number of clusters in advance. |
| **Simple:** Easy to implement, understand, and interpret the cluster centers. | **Spherical Assumption:** Assumes clusters are spherical and evenly sized. Fails on nested/moon-shaped data. |
| **Guaranteed Convergence:** It will always find a solution (though it might be a local minimum). | **Sensitive to Outliers:** Centroids are pulled heavily by outliers (since it uses the mean). |
| **Great Baseline:** Usually the first clustering algorithm you should try on a new dataset. | **Needs Scaling:** Because it relies on Euclidean distance, features *must* be on the same scale. |

## Evaluation Metrics
Since there are no "true" labels to check against, evaluating clustering is tricky:
*   **Inertia (WCSS):** Lower is better, but naturally decreases as $k$ increases.
*   **Silhouette Score:** Ranges from -1 to 1. Measures how similar a point is to its own cluster compared to other clusters. Values near 1 indicate dense, well-separated clusters.

## Real-World Projects & Use Cases
1.  **Customer Segmentation:** Grouping retail customers based on purchasing behavior, age, and income for targeted marketing campaigns.
2.  **Image Compression:** Reducing the number of colors in an image. If you run K-Means with $k=16$ on image pixels, you can compress a full-color image down to just 16 colors (the centroids).
3.  **Document Clustering:** Grouping similar articles or web pages together based on word frequencies.
4.  **Anomaly Detection:** If a point is exceptionally far from its assigned centroid, it might be an anomaly or fraud.

## Sample Python Implementation (scikit-learn)

```python
# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# 2. Load Data (Example)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# 3. SCALE THE FEATURES (Crucial for K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Find optimal K using the Elbow Method
wcss = []
for i in range(1, 11):
    # init='k-means++' prevents bad random initialization traps
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# (Optional) Plot the Elbow Graph
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('WCSS')
# plt.show()

# 5. Train the final model with optimal K (let's assume k=4 from the elbow)
optimal_k = 4
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
cluster_labels = final_kmeans.fit_predict(X_scaled)

# 6. Get the cluster centers
centroids = final_kmeans.cluster_centers_

print(f"Cluster Centers (Scaled):\n{centroids}")
print(f"First 10 cluster assignments: {cluster_labels[:10]}")
