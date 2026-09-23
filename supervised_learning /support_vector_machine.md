# Support Vector Machine (SVM) Cheatsheet

**Type:** Supervised Learning (Classification & Regression)  
**Output:** Discrete (Class label - SVC) or Continuous (Value - SVR)

## Intuition
Imagine you have red dots and blue dots on a piece of paper, and you want to draw a straight line to separate them. You could draw infinite lines, but the **Support Vector Machine (SVM)** aims to draw the *best* line. 

The "best" line (called the **Hyperplane**) is the one that has the maximum distance (the **Margin**) between the line and the closest data points of both classes. These closest, most critical data points that dictate where the line goes are called the **Support Vectors**. If you remove all other data points, the hyperplane wouldn't move—only the support vectors matter!

## The Math (Linear Algebra at its Core)
For a linearly separable dataset, the hyperplane is defined as:
$$w \cdot x - b = 0$$
*(where $w$ is the normal vector to the hyperplane, $x$ is the input vector, and $b$ is the bias).*

The margins are defined by the equations:
*   Positive class boundary: $w \cdot x - b = 1$
*   Negative class boundary: $w \cdot x - b = -1$

The goal is to maximize the margin, which mathematically equates to minimizing $\vert{}\vert{}w\vert{}\vert{}$.

## The Kernel Trick (Handling Non-Linearity)
What if the red and blue dots are arranged in a circle, and a straight line *cannot* separate them? 
SVM uses the **Kernel Trick**. It mathematically transforms the 2D space into a 3D space (or higher) where a flat 2D plane *can* slice between the classes. It does this without actually computing the coordinates in the higher dimension, making it computationally efficient.

**Common Kernels:**
1.  **Linear:** Good for simple, linearly separable data.
2.  **Polynomial:** Good for curved boundaries.
3.  **RBF (Radial Basis Function):** The default and most powerful. Good for circular, complex boundaries.
4.  **Sigmoid:** Used similarly to neural networks.

## Key Hyperparameters
*   **`C` (Regularization):** Controls the tradeoff between a smooth decision boundary and classifying training points correctly.
    *   *High C:* "Hard Margin" - Strict. Tries to classify every point perfectly (Risk of Overfitting).
    *   *Low C:* "Soft Margin" - Forgiving. Allows some misclassifications for a broader, more generalized boundary.
*   **`gamma` (Kernel Coefficient):** Determines how far the influence of a single training point reaches (used in RBF/Poly).
    *   *High gamma:* Points must be very close to influence each other. Boundary becomes jagged and clings to points (Overfitting).
    *   *Low gamma:* Influence reaches far. Boundary becomes smoother (Underfitting).
*   **`kernel`:** Which kernel to use (`'linear'`, `'rbf'`, `'poly'`).

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Highly Effective:** Excels in high-dimensional spaces (even when dimensions > number of samples). | **Computationally Expensive:** Training time scales cubically with the size of the dataset (bad for massive datasets). |
| **Memory Efficient:** Uses only a subset of training points (support vectors) to make decisions. | **Hyperparameter Sensitivity:** Requires careful tuning of `C`, `gamma`, and the `kernel` to get good results. |
| **Versatile:** The Kernel Trick allows it to solve highly complex, non-linear problems. | **Not Interpretable:** The mathematical transformations make it a "black box" model. |
| **Robust to Outliers:** (When using a soft margin / appropriate `C` value). | **Feature Scaling Required:** Distance calculations mean normalization/standardization is mandatory. |

## Evaluation Metrics
*   **Classification (SVC):** Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
*   **Regression (SVR):** Mean Squared Error (MSE), R-Squared.

## Real-World Projects & Use Cases
1.  **Image Recognition:** Classifying images (e.g., face detection or handwriting recognition) where the number of pixels (dimensions) is extremely high.
2.  **Bioinformatics:** Protein fold and cancer tissue classification where datasets have thousands of genes but only dozens of patients.
3.  **Text Categorization:** Classifying news articles or identifying spam (though Naive Bayes is usually faster for this).
4.  **Geospatial Analysis:** Predicting land use from satellite imagery.

## Sample Python Implementation (Scikit-learn)

```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 2. Load data (Example)
# df = pd.read_csv('cancer_data.csv')
# X = df.drop('malignant', axis=1)
# y = df['malignant']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. SCALE THE FEATURES (Mandatory for SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and train the model
# Using RBF kernel by default
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Make predictions
y_pred = model.predict(X_test_scaled)

# 7. Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
