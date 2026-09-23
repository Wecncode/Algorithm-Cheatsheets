# Handling Missing Data Cheatsheet

**Type:** Data Preprocessing  
**Output:** A complete dataset with no `NaN` or `Null` values, ready for machine learning.

## Intuition
Real-world data is messy. Sensors fail, users skip questions on surveys, and data gets corrupted. Most Machine Learning algorithms (like Linear Regression, SVM, and Neural Networks) will immediately crash if they encounter a missing value (`NaN`). 

Before training a model, you must either **remove** the missing data or **guess** (impute) what the missing data should have been.

## Types of Missing Data
Understanding *why* the data is missing dictates how you handle it:
1.  **MCAR (Missing Completely At Random):** There is no pattern to the missingness. (e.g., A wind gust blew away a random page of a survey). 
2.  **MAR (Missing At Random):** The missingness is related to *other* observed variables, but not the missing value itself. (e.g., Men might be more likely to skip a question about depression, but it's not because of their depression level).
3.  **MNAR (Missing Not At Random):** The missingness is directly related to the value that is missing. (e.g., Extremely wealthy people refusing to answer a question about their income). *This is the hardest to handle.*

## The Core Techniques

### 1. Deletion
*   **Row Deletion (Listwise):** Drop any row that has at least one missing value.
    *   *Pros:* Easy, keeps data distribution pure.
    *   *Cons:* Can drastically reduce dataset size. Only use if missing data is < 5% of the total dataset.
*   **Column Deletion:** Drop the entire feature/column.
    *   *Pros:* Solves the problem instantly.
    *   *Cons:* Destroys potentially valuable information. Only use if a column is missing > 50% of its data.

### 2. Simple Imputation
*   **Mean Imputation:** Replace missing values with the average of the column. (Use for normally distributed numerical data without outliers).
*   **Median Imputation:** Replace missing values with the middle value. (Use for skewed numerical data or data with extreme outliers).
*   **Mode Imputation:** Replace missing values with the most frequent value. (Use for Categorical data).
*   **Constant Imputation:** Replace with a placeholder like "Unknown" or `0`. (Good for highlighting that the data *was* missing, which might be a signal itself).

### 3. Advanced Imputation (Model-Based)
*   **KNN Imputer:** Looks at the $k$ most similar rows (neighbors) and averages their values to fill in the blank.
*   **Iterative Imputer (MICE):** Treats the missing column as the target variable ($y$) and uses all other columns ($X$) to run a regression model to predict the missing value. 

## Decision Matrix: What should you do?

| Scenario | Recommendation |
| :--- | :--- |
| **Missing < 5% of rows (MCAR)** | **Row Deletion.** Just drop them, it won't hurt the model. |
| **Column is > 50% empty** | **Column Deletion.** Drop the feature entirely. |
| **Numerical Data (No Outliers)** | **Mean Imputation.** Fast and mathematically sound. |
| **Numerical Data (Heavy Outliers)**| **Median Imputation.** Outliers will heavily skew the mean. |
| **Categorical / Text Data** | **Mode Imputation** or create a new category called "Missing". |
| **Small Dataset / High Accuracy Needed** | **KNN or Iterative Imputer.** Recovers the most accurate estimates. |

## The Golden Rule: Prevent Data Leakage!
Just like feature scaling, **you MUST fit your Imputer on the Training Data ONLY.**
Calculate the mean/median from `X_train`, and use *that exact same number* to fill missing values in `X_test`. Never calculate the mean of the entire dataset before splitting!

## Sample Python Implementation (Pandas & scikit-learn)

```python
# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

# 2. Create Example Data with Missing Values
data = {
    'Age': [25, np.nan, 30, 45, 35],             # Numerical (Normal)
    'Income': [50000, 60000, np.nan, 2000000, 55000], # Numerical (Outlier!)
    'City': ['NY', 'LA', 'NY', np.nan, 'SF']     # Categorical
}
df = pd.DataFrame(data)
print("Original Data with NaNs:\n", df, "\n")

# SPLIT FIRST! Never impute before splitting.
# (Assuming a dummy target variable 'y' exists)
y = [0, 1, 0, 1, 1] 
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# ==========================================
# APPROACH 1: PANDAS DELETION (Quick & Dirty)
# ==========================================
# Drop rows with any NaN
df_dropped_rows = df.dropna(axis=0) 

# Drop columns with > 50% missing data
df_dropped_cols = df.dropna(axis=1, thresh=int(0.5 * len(df))) 


# ==========================================
# APPROACH 2: SIMPLE IMPUTER (Scikit-Learn)
# ==========================================
# A. Impute 'Age' with Mean
mean_imputer = SimpleImputer(strategy='mean')
X_train['Age'] = mean_imputer.fit_transform(X_train[['Age']])
# X_test['Age'] = mean_imputer.transform(X_test[['Age']]) # Example of test set transform

# B. Impute 'Income' with Median (because of the 2M outlier)
median_imputer = SimpleImputer(strategy='median')
X_train['Income'] = median_imputer.fit_transform(X_train[['Income']])

# C. Impute 'City' with Mode (Most Frequent)
mode_imputer = SimpleImputer(strategy='most_frequent')
X_train['City'] = mode_imputer.fit_transform(X_train[['City']])


# ==========================================
# APPROACH 3: ADVANCED KNN IMPUTATION
# ==========================================
# Note: KNN Imputer only works on numerical data! You must encode categorical data first.
numeric_df = df[['Age', 'Income']]

knn_imputer = KNNImputer(n_neighbors=2)
imputed_array = knn_imputer.fit_transform(numeric_df)
df_knn_imputed = pd.DataFrame(imputed_array, columns=numeric_df.columns)

print("Data after KNN Imputation:\n", df_knn_imputed)**
