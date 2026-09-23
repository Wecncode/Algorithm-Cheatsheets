# Evaluation Metrics Cheatsheet

**Type:** Model Evaluation  
**Purpose:** Quantifying how well your Machine Learning model performs on unseen data.

## Intuition
Training a model is only half the battle. If a model predicts that 99% of transactions are legitimate, and only 1% are fraud, a "dumb" model that simply guesses "Legitimate" every single time will be 99% accurate! But it completely failed its purpose. 

Choosing the *right* evaluation metric ensures you are measuring what actually matters for your specific business problem.

---

## Classification Metrics
*Used when predicting discrete labels (e.g., Spam vs. Inbox, Cat vs. Dog).*

### 1. The Confusion Matrix
The foundation of classification metrics. It breaks down predictions into four categories:
*   **True Positive (TP):** Predicted Positive, and it *is* Positive.
*   **True Negative (TN):** Predicted Negative, and it *is* Negative.
*   **False Positive (FP - Type I Error):** Predicted Positive, but it is actually Negative (e.g., sending a real email to the Spam folder).
*   **False Negative (FN - Type II Error):** Predicted Negative, but it is actually Positive (e.g., missing a fraudulent transaction).

### 2. Accuracy
The percentage of total predictions that were correct.
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
*   **When to use:** When your classes are perfectly balanced (e.g., 50% cats, 50% dogs). 
*   **When to avoid:** Imbalanced datasets (e.g., 99% healthy, 1% sick).

### 3. Precision
Out of all the instances the model *predicted* as positive, how many were *actually* positive? 
$$Precision = \frac{TP}{TP + FP}$$
*   **When to use:** When **False Positives are highly costly**. (e.g., Spam filters. You don't want to miss an important email from your boss by falsely labeling it as spam).

### 4. Recall (Sensitivity / True Positive Rate)
Out of all the *actual* positive instances in the data, how many did the model *find*?
$$Recall = \frac{TP}{TP + FN}$$
*   **When to use:** When **False Negatives are highly costly**. (e.g., Cancer detection. It is better to falsely flag a healthy patient for a second screening than to send a sick patient home).

### 5. F1-Score
The harmonic mean of Precision and Recall. It punishes extreme values, providing a single balanced metric.
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
*   **When to use:** When you have an imbalanced dataset and need a balance between Precision and Recall.

### 6. ROC-AUC
The Area Under the Receiver Operating Characteristic Curve. It measures the model's ability to distinguish between classes across all possible probability thresholds.
*   **AUC = 0.5:** Random guessing.
*   **AUC = 1.0:** Perfect model.
*   **When to use:** Evaluating the overall ranking performance of a binary classifier, regardless of the strict 0.5 threshold.

---

## Regression Metrics
*Used when predicting continuous numbers (e.g., House Prices, Temperature).*

### 1. Mean Absolute Error (MAE)
The average of the absolute differences between the predicted values and the actual values.
$$MAE = \frac{1}{n} \sum \vert{}y_{actual} - y_{predicted}\vert{}$$
*   **Pros:** Highly interpretable. (e.g., "Our house price predictions are off by $5,000 on average"). Robust to outliers.

### 2. Mean Squared Error (MSE)
The average of the *squared* differences between predictions and actuals.
$$MSE = \frac{1}{n} \sum (y_{actual} - y_{predicted})^2$$
*   **Pros:** Punishes large errors severely (because they are squared).
*   **Cons:** Not interpretable in the original units (e.g., "squared dollars").

### 3. Root Mean Squared Error (RMSE)
The square root of the MSE.
$$RMSE = \sqrt{MSE}$$
*   **Pros:** Punishes large errors AND is interpretable in the original units. This is the most popular default regression metric.

### 4. R-Squared ($R^2$ - Coefficient of Determination)
Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
*   **$R^2 = 1.0$:** Perfect predictions.
*   **$R^2 = 0.0$:** The model is no better than just guessing the average value every time.
*   **Pros:** Gives a percentage of "goodness of fit" that is easy to explain to stakeholders.

---

## Sample Python Implementation (scikit-learn)

```python
# Import metrics libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ==========================================
# 1. CLASSIFICATION METRICS
# ==========================================
# y_true = actual labels, y_pred = model's hard predictions, y_prob = model probabilities
y_true_cls = [0, 1, 1, 0, 1, 0]
y_pred_cls = [0, 1, 0, 0, 1, 1]
y_prob_cls = [0.1, 0.9, 0.4, 0.2, 0.8, 0.7]

print("--- CLASSIFICATION ---")
print("Accuracy:", accuracy_score(y_true_cls, y_pred_cls))
print("Precision:", precision_score(y_true_cls, y_pred_cls))
print("Recall:", recall_score(y_true_cls, y_pred_cls))
print("F1-Score:", f1_score(y_true_cls, y_pred_cls))
print("ROC-AUC:", roc_auc_score(y_true_cls, y_prob_cls))

print("\nConfusion Matrix:\n", confusion_matrix(y_true_cls, y_pred_cls))
print("\nClassification Report:\n", classification_report(y_true_cls, y_pred_cls))

# ==========================================
# 2. REGRESSION METRICS
# ==========================================
y_true_reg = [100, 150, 200, 250, 300]
y_pred_reg = [110, 140, 210, 240, 320]

print("\n--- REGRESSION ---")
print("MAE:", mean_absolute_error(y_true_reg, y_pred_reg))
print("MSE:", mean_squared_error(y_true_reg, y_pred_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))
print("R-Squared:", r2_score(y_true_reg, y_pred_reg))
