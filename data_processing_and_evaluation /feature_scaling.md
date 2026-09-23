# Feature Scaling Cheatsheet
*(Normalization vs. Standardization)*

**Type:** Data Preprocessing  
**Output:** Transformed numerical features ready for machine learning algorithms.

## Intuition
Imagine you are predicting house prices. One feature is "Number of Bedrooms" (ranging from 1 to 5). Another feature is "Square Footage" (ranging from 500 to 5,000). 

Many machine learning algorithms calculate the *distance* between data points or use *Gradient Descent*. If you don't scale your data, the "Square Footage" feature will completely dominate the "Number of Bedrooms" simply because the numbers are bigger. The algorithm will think square footage is 1,000 times more important, which is a mathematical illusion.

**Feature scaling** forces all numeric features onto a level playing field, ensuring the algorithm judges them based on their actual patterns, not their raw magnitude.

## The Core Techniques

### 1. Standardization (Z-Score Scaling)
*   **What it does:** Centers the data around a mean of `0` and a standard deviation of `1`. 
*   **The Math:** 
    $$z = \frac{x - \mu}{\sigma}$$ 
    *(where $\mu$ is the mean and $\sigma$ is the standard deviation)*
*   **Result:** Data will mostly fall between -3 and +3, but it does *not* strictly bound the data. It preserves the shape of outliers.

### 2. Normalization (Min-Max Scaling)
*   **What it does:** Squashes the data into a strict, fixed range, usually exactly `0` to `1`.
*   **The Math:**
    $$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
*   **Result:** The absolute lowest value becomes 0, the highest becomes 1. Highly sensitive to extreme outliers (one massive outlier will squish all normal data into a tiny 0.001 range).

## Which one should you use?

| Scenario | Recommendation |
| :--- | :--- |
| **Default / Not Sure?** | **Standardization.** It is generally the safest bet and handles outliers much better. |
| **Distance-Based Algorithms** (KNN, SVM, K-Means) | **Standardization.** Ensures all axes are weighted equally when measuring Euclidean distance. |
| **Gradient Descent Algorithms** (Neural Networks, Linear/Logistic Regression) | **Standardization.** Helps the optimizer converge much faster by making the loss landscape circular instead of elliptical. |
| **Image Processing (Computer Vision)** | **Normalization.** Pixel values are strictly bounded (0 to 255), so dividing by 255 perfectly normalizes them to 0-1. |
| **Algorithms that assume NO distribution** (e.g., strictly positive data bounds) | **Normalization.** When you absolutely need your data to be within a specific bounding box (like probabilities). |
| **Tree-Based Algorithms** (Decision Trees, Random Forest, XGBoost) | **NEITHER!** Tree algorithms split data based on thresholds (e.g., $x > 500$). They don't care about distance or magnitude. Scaling is unnecessary. |

## The Golden Rule: Prevent Data Leakage!
**You MUST fit your scaler on the Training Data ONLY.**
1.  `fit_transform` the `X_train` data (this calculates the min/max or mean/std).
2.  ONLY `transform` the `X_test` data using those saved calculations. 

Tip: *If you fit the scaler on your entire dataset before splitting, information from the test set "leaks" into the training set, ruining your evaluation!*

## Python Implementation (scikit-learn)

```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 2. Load and Split Data
# df = pd.read_csv('housing_data.csv')
# X = df[['bedrooms', 'square_footage', 'age']]
# y = df['price']

# SPLIT FIRST! Never scale before splitting.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# APPROACH A: STANDARDIZATION (Recommended)
# ==========================================
std_scaler = StandardScaler()

# Fit on training data AND transform it
X_train_std = std_scaler.fit_transform(X_train)

# ONLY transform the test data (using training mean/std)
X_test_std = std_scaler.transform(X_test)


# ==========================================
# APPROACH B: NORMALIZATION (Min-Max)
# ==========================================
minmax_scaler = MinMaxScaler()

# Fit on training data AND transform it
X_train_norm = minmax_scaler.fit_transform(X_train)

# ONLY transform the test data (using training min/max)
X_test_norm = minmax_scaler.transform(X_test)

# Example output to verify
print("Standardized Mean (should be ~0):", X_train_std[:, 0].mean())
print("Standardized Std (should be ~1):", X_train_std[:, 0].std())

print("\nNormalized Min (should be 0):", X_train_norm[:, 0].min())
print("Normalized Max (should be 1):", X_train_norm[:, 0].max())
