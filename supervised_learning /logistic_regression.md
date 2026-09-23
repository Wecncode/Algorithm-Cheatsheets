# Logistic Regression Cheatsheet

**Type:** Supervised Learning (Classification)  
**Output:** Discrete / Categorical (e.g., 0 or 1, True or False)

## Intuition
Despite its name, Logistic Regression is a **classification** algorithm, not a regression algorithm. It is used to predict the probability that an instance belongs to a given class. If the estimated probability is greater than 50%, the model predicts that the instance belongs to that class (called the positive class, labeled "1").

It works by passing a linear regression equation through a **Sigmoid (Logistic) Function**, which squashes the output values to strictly fall between 0 and 1.

## The Math

**1. The Linear Equation:**  
$$z = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n$$

**2. The Sigmoid Function:**  
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**3. The Final Prediction:**  
$$P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... + \beta_nX_n)}}$$

*If $P(Y=1) \ge 0.5$, predict 1. Else, predict 0.*

## Types of Logistic Regression
1. **Binary:** Only two possible outcomes (e.g., Spam or Not Spam).
2. **Multinomial:** Three or more categories without ordering (e.g., predicting animal type: Cat, Dog, or Bird).
3. **Ordinal:** Three or more categories with ordering (e.g., Movie ratings from 1 to 5 stars).

## Key Assumptions
* **Independent Observations:** Observations are independent of each other.
* **Little/No Multicollinearity:** Independent variables should not be highly correlated.
* **Linearity of Independent Variables and Log-Odds:** The independent variables are linearly related to the log-odds of the dependent variable.
* **Large Sample Size:** Typically requires a larger sample size than linear regression.

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| Highly interpretable (coefficients represent odds ratios). | Struggles with non-linear relationships. |
| Very fast to train and predict. | Prone to overfitting with high-dimensional data (needs regularization). |
| Outputs well-calibrated probabilities, not just hard classifications. | Requires independent variables to be linearly related to log-odds. |
| Performs exceptionally well on linearly separable data. | Outliers can significantly skew the results. |

## Evaluation Metrics
Since this is a classification task, we do not use RMSE or R-squared. We use:
* **Confusion Matrix:** Shows True Positives, True Negatives, False Positives, False Negatives.
* **Accuracy:** Overall correctness ($\frac{TP+TN}{Total}$).
* **Precision:** Out of all positive *predictions*, how many were right? ($\frac{TP}{TP+FP}$)
* **Recall (Sensitivity):** Out of all *actual* positives, how many did we find? ($\frac{TP}{TP+FN}$)
* **F1-Score:** Harmonic mean of Precision and Recall. Good for imbalanced datasets.
* **ROC-AUC:** Measures the model's ability to distinguish between classes.

## Real-World Projects & Use Cases
1. **Email Spam Filter:** Predicting whether an incoming email is "Spam" (1) or "Inbox" (0) based on word frequencies and sender metadata.
2. **Credit Risk Scoring:** Banks use it to predict the probability of a customer defaulting on a loan (Default vs. No Default) based on income, credit history, and employment.
3. **Medical Diagnosis:** Predicting the presence or absence of a disease (e.g., Diabetes) based on patient metrics like BMI, blood pressure, and age.
4. **Customer Churn Prediction:** Determining whether a subscriber is likely to cancel their subscription service in the next month.

## Sample Python Implementation (scikit-learn)

```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load data (Example)
# df = pd.read_csv('credit_data.csv')
# X = df[['income', 'credit_score', 'loan_amount']]
# y = df['default_status']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the model
# max_iter is increased for convergence on standard datasets
model = LogisticRegression(max_iter=1000) 
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Get probabilities for the positive class

# 6. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
