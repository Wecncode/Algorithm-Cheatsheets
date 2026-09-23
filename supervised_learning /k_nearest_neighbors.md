# K-Nearest Neighbors (KNN) Cheatsheet

**Type:** Supervised Learning (Classification & Regression)  
**Output:** Discrete (Class label) or Continuous (Average value)

## Intuition
KNN is often called a **"lazy learner"** because it doesn't actually "train" a model or learn a mathematical function in advance. Instead, it memorizes the entire training dataset. 

When a new, unseen data point arrives, the algorithm looks at the **$k$** closest data points (neighbors) in the training set. 
*   **For Classification:** It takes a majority vote (e.g., if 3 of the 5 closest neighbors are "Cats", the new point is classified as a "Cat").
*   **For Regression:** It calculates the average value of those $k$ closest neighbors.

## The Math (Distance Metrics)
To find the "closest" neighbors, we need a way to measure distance. 

**1. Euclidean Distance (Most Common - L2 Norm):**  
The straight-line distance between two points ($p$ and $q$).
$$d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$

**2. Manhattan Distance (L1 Norm):**  
The distance between two points measured along axes at right angles (like a taxi driving through city blocks).
$$d(p, q) = \sum_{i=1}^{n} \vert{}q_i - p_i\vert{}$$

**3. Minkowski Distance:**  
A generalized mathematical form where $p=1$ is Manhattan and $p=2$ is Euclidean.
$$d(p, q) = \left( \sum_{i=1}^{n} \vert{}q_i - p_i\vert{}^p \right)^{1/p}$$

## The 'K' Hyperparameter
Choosing the right $k$ is critical:
*   **Small $k$ (e.g., $k=1$):** High variance, low bias. Very sensitive to noise/outliers (Overfitting).
*   **Large $k$ (e.g., $k=50$):** Low variance, high bias. Smoother decision boundaries but might ignore subtle patterns (Underfitting).
*   **Pro-Tip:** Always choose an **odd number** for $k$ in binary classification to prevent tie votes!

## Crucial Assumption: Feature Scaling
**Tip**: Because KNN relies purely on distance, **you MUST scale your data** (e.g., using Normalization or Standardization). If Feature A ranges from 0-1 and Feature B ranges from 0-10,000, Feature B will completely dominate the distance calculation, rendering Feature A useless.
*   **Feature Scaling is Mandatory:** Because KNN relies on distance, features with larger ranges (e.g., salary in thousands) will completely overpower features with smaller ranges (e.g., age in decades). You **must** normalize or standardize your data first.
*   **Curse of Dimensionality:** KNN struggles in high-dimensional spaces. As the number of features grows, the concept of "distance" becomes less meaningful, and points become essentially equidistant.

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| Very simple to understand and implement. | **Slow during prediction:** Has to calculate the distance to *every* training point. |
| No training phase (instantly ready to predict). | **Memory intensive:** Requires storing the entire dataset in RAM. |
| Naturally handles multi-class classification. | **Curse of Dimensionality:** Performance degrades heavily with too many features. |
| Makes no assumptions about data distribution (non-parametric). | Highly sensitive to outliers and imbalanced datasets. |

## Evaluation Metrics
*   **Classification:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
*   **Regression:** RMSE, MAE, R-Squared.

## Real-World Projects & Use Cases
1.  **Basic Recommender Systems:** Finding users with similar tastes (e.g., "Users who watched this also watched...").
2.  **Missing Data Imputation:** Guessing the value of a missing cell in a dataset based on the values of the most similar rows (`KNNImputer` in scikit-learn).
3.  **Anomaly Detection:** Detecting credit card fraud. If a transaction is plotted and its nearest neighbors are very far away, it might be an anomaly.
4.  **Handwriting Recognition:** Classifying drawn digits based on pixel similarity to known digits.

## Sample Python Implementation (Scikit-learn)

```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Load data (Example)
# df = pd.read_csv('customer_data.csv')
# X = df[['age', 'annual_income', 'spending_score']]
# y = df['purchased']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. SCALE THE FEATURES (Crucial for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and train the model (k=5 is default)
k = 5
model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
model.fit(X_train_scaled, y_train)

# 6. Make predictions
y_pred = model.predict(X_test_scaled)

# 7. Evaluate the model
print(f"Accuracy with k={k}: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
