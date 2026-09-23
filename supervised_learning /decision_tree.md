# Decision Tree Cheatsheet

**Type:** Supervised Learning (Classification & Regression)  
**Output:** Discrete (Class label - CART/ID3) or Continuous (Average value - Regression Tree)

## Intuition
A Decision Tree is a flowchart-like model that makes predictions by asking a series of True/False questions about the features. 

Starting at the **Root Node** (the entire dataset), it finds the feature and the threshold that best splits the data into two distinct, homogeneous groups (purest possible). It repeats this recursively for each branch until it reaches a stopping criterion, ending at the **Leaf Nodes** (the final predictions).

Because they create step-like, orthogonal splits in the feature space, they are incredibly intuitive and mirror human decision-making.

## The Math (Splitting Criteria)
How does the tree decide which question to ask? It tries to minimize **Impurity**.

**1. Entropy (Information Theory):**  
Measures the amount of randomness or disorder in a node. (0 = perfectly pure, 1 = maximum disorder).
$$H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$$
*(where $p_i$ is the probability of class $i$ in node $S$)*

**2. Information Gain:**  
The reduction in Entropy after splitting a node on feature $A$. The tree chooses the split that maximizes this.
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{\vert{}S_v\vert{}}{\vert{}S\vert{}} H(S_v)$$

**3. Gini Impurity (Default in scikit-learn):**  
Measures the probability of misclassifying a randomly chosen element. Slightly faster to compute than Entropy because it avoids logarithms.
$$Gini = 1 - \sum_{i=1}^{c} p_i^2$$

## Key Hyperparameters
*   **`max_depth`:** The maximum number of levels in the tree. (Crucial for preventing overfitting).
*   **`min_samples_split`:** The minimum number of samples required to split an internal node.
*   **`min_samples_leaf`:** The minimum number of samples required to be at a leaf node.

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Highly Interpretable:** Easy to visualize and explain to non-technical stakeholders. | **Prone to Overfitting:** Without constraints, trees will perfectly memorize the training data. |
| **No Preprocessing Required:** Does not require feature scaling (normalization/standardization). | **High Variance:** A tiny change in the training data can result in a completely different tree. |
| Can handle both numerical and categorical data natively. | Biased toward features with many levels/categories. |
| Naturally captures non-linear relationships and feature interactions. | Creates rigid, jagged decision boundaries (orthogonal). |

## Evaluation Metrics
*   **Classification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.
*   **Regression:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R-Squared.

## Real-World Projects & Use Cases
1.  **Medical Diagnosis:** Triage systems (e.g., "Is patient temperature > 101? If yes -> Is blood pressure > 140?").
2.  **Customer Churn:** Predicting whether a user will cancel a subscription based on usage frequency, account age, and support tickets.
3.  **Credit Risk:** Simple rule-based approval for small loans based on income thresholds and credit history.
4.  **Manufacturing Fault Diagnosis:** Identifying the root cause of machinery failure based on sensor readings.

## Sample Python Implementation (Scikit-learn)

```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 2. Load data (Example)
# df = pd.read_csv('loan_data.csv')
# X = df[['income', 'loan_amount', 'credit_score']]
# y = df['approved']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the model
# Constraining max_depth is vital to prevent overfitting!
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Bonus: Visualize the tree!
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, class_names=['Denied', 'Approved'], filled=True)
plt.show()
