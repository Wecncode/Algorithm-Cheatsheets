# Boosting (AdaBoost, GBM, XGBoost) Cheatsheet

**Type:** Supervised Learning (Classification & Regression)  
**Output:** Discrete (Class label) or Continuous (Value)

## Intuition
While Random Forest is about building many independent trees at the same time (Bagging), **Boosting** is all about learning from past mistakes. 

Boosting trains models (usually shallow decision trees called "weak learners") **sequentially**. The first tree tries to predict the target. It will inevitably get some predictions wrong. The *second* tree is then explicitly trained to focus on correcting the mistakes made by the first tree. The third tree corrects the second, and so on.

By combining all these weak, focused learners together, you get a single, incredibly strong predictive model.

## The Core Variants

**1. AdaBoost (Adaptive Boosting):**
*   **How it works:** After the first tree makes its predictions, AdaBoost increases the "weight" (importance) of the data points that were misclassified. 
*   The next tree is forced to pay more attention to these heavier, harder-to-predict points.
*   Final prediction is a weighted vote (better trees get more say).

**2. Gradient Boosting Machines (GBM):**
*   **How it works:** Instead of tweaking data weights, GBM calculates the **Residuals** (the exact mathematical difference between the actual value and the predicted value).
*   The next tree is trained to predict the *residual errors* of the previous tree, not the actual target variable! 
*   It uses Gradient Descent to minimize the loss function as trees are added.

**3. XGBoost (Extreme Gradient Boosting):**
*   The undisputed king of competitive machine learning (Kaggle). 
*   It is a heavily optimized, mathematically refined version of GBM. It includes built-in L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting, uses advanced hardware optimization (parallel processing for tree building), and handles missing data automatically.

## Key Hyperparameters (XGBoost / GBM)
*   **`n_estimators`:** The number of trees (boosting rounds). Too high = overfitting.
*   **`learning_rate` (eta):** Shrinks the contribution of each new tree. A lower learning rate requires more `n_estimators` but makes the model highly robust and generalizable.
*   **`max_depth`:** The maximum depth of the individual trees. Usually kept shallow in boosting (e.g., 3 to 7) because we want weak learners.
*   **`subsample`:** The fraction of observations to randomly sample for each tree (adds a bagging-like effect to prevent overfitting).

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Best-in-Class Accuracy:** Often outperforms all other traditional ML algorithms on tabular (structured) data. | **Prone to Overfitting:** Unlike Random Forest, adding too many trees *will* cause boosting to overfit. |
| **Handles Imbalanced Data:** Easily customized to penalize false negatives/positives via custom loss functions. | **Sensitive to Outliers:** Because it focuses on errors, severe outliers can derail the model's focus. |
| **Feature Importance:** Provides clear insights into which variables drive the predictions. | **Hyperparameter Heavy:** Requires careful tuning of learning rate, depth, and estimators to work well. |
| **Missing Values (XGBoost):** XGBoost handles missing data automatically without requiring prior imputation. | **Slower Training:** Sequential nature means it cannot be entirely parallelized like a Random Forest. |

## Evaluation Metrics
*   **Classification:** ROC-AUC (very common for evaluating boosting models), F1-Score, Log-Loss (Cross-Entropy).
*   **Regression:** RMSE, MAE, R-Squared.

## Real-World Projects & Use Cases
1.  **Search Engine Ranking:** Algorithms like RankBrain use gradient boosting to sort and rank search results based on relevance.
2.  **Dynamic Pricing:** Airlines and ride-sharing apps (like Uber) use XGBoost to calculate real-time prices based on demand, weather, and time.
3.  **Customer Conversion Prediction:** E-commerce platforms predicting which users are most likely to buy an item in their cart.
4.  **Insurance Risk:** Predicting the exact financial loss or risk severity of an insurance policyholder.

## Sample Python Implementation (XGBoost)

*Note: You may need to run `pip install xgboost` first.*

```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# 2. Load data (Example)
# df = pd.read_csv('conversion_data.csv')
# X = df.drop('converted', axis=1)
# y = df['converted']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the model
# Using common, robust hyperparameters
model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    subsample=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Bonus: Plot Feature Importance
import matplotlib.pyplot as plt
xgb.plot_importance(model, max_num_features=10)
plt.show()
