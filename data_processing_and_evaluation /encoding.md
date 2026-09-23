# Encoding Categorical Data Cheatsheet

**Type:** Data Preprocessing  
**Output:** Numerical representations of text/categorical data ready for ML algorithms.

## Intuition
Machine Learning models are fundamentally giant calculators. They perform math (addition, multiplication, gradients) on data. You cannot multiply the color "Red" by a weight of `0.5`. 

Therefore, before feeding data into almost any algorithm (except for some tree-based models like CatBoost or LightGBM), you **must** convert all text-based categorical data into numbers. This process is called **Encoding**.

## The Core Techniques

### 1. Ordinal Encoding (Label Encoding)
*   **How it works:** Assigns a unique integer to each category (e.g., `Small` = 1, `Medium` = 2, `Large` = 3).
*   **The Danger:** By assigning numbers, you are mathematically telling the algorithm that there is an **order** and **magnitude**. The model will assume that `Large` is three times bigger/better than `Small`. 
*   **When to use:** ONLY use this for **Ordinal Data** (categories that have a natural, logical order like T-shirt sizes, education levels, or satisfaction ratings).

### 2. One-Hot Encoding (OHE / Dummy Variables)
*   **How it works:** Creates a brand new binary column (0 or 1) for *every* unique category in the original column.
*   **The Dummy Variable Trap:** If you have a column for `Male` and `Female`, knowing it's not Male guarantees it's Female. This creates perfect multicollinearity, which breaks linear models (Linear/Logistic Regression). **Fix:** Always drop one of the created columns (e.g., `drop_first=True`).
*   **When to use:** Use for **Nominal Data** (categories with no inherent order, like Colors, Cities, or Car Brands).

### 3. Target Encoding (Mean Encoding) - *Advanced*
*   **How it works:** Replaces a category with the average value of the *target variable* for that specific category. (e.g., If the historical average house price in "Brooklyn" is $800k, replace "Brooklyn" with 800000).
*   **The Danger:** Severe risk of **Data Leakage** and overfitting if not done carefully using cross-validation techniques.
*   **When to use:** When you have **High Cardinality** data (Nominal data with hundreds or thousands of unique categories, like Zip Codes). One-Hot Encoding a Zip Code column would create 10,000 new columns and crash your model!

## Decision Matrix: Which one to choose?

| Data Type | Example | Best Encoding Technique |
| :--- | :--- | :--- |
| **Ordinal** (Has a logical order) | Low, Medium, High | **Ordinal Encoding** |
| **Nominal - Low Cardinality** (< 15 unique values) | Red, Green, Blue | **One-Hot Encoding** |
| **Nominal - High Cardinality** (> 15 unique values) | Zip Codes, User IDs, IP Addresses | **Target Encoding** (or drop the column) |
| **Target Variable (y)** | "Yes" or "No", "Spam" or "Inbox" | **Label Encoding** (0 and 1) |

## Sample Python Implementation (Pandas & scikit-learn)

```python
# 1. Import libraries
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

# 2. Create Example Data
data = {
    'Size': ['Small', 'Large', 'Medium', 'Small', 'Large'], # Ordinal
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],       # Nominal
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes']          # Target
}
df = pd.DataFrame(data)

# ==========================================
# 1. ORDINAL ENCODING (For Features)
# ==========================================
# We must explicitly define the order so it doesn't just do it alphabetically
size_order = [['Small', 'Medium', 'Large']]
ordinal_encoder = OrdinalEncoder(categories=size_order)

df['Size_Encoded'] = ordinal_encoder.fit_transform(df[['Size']])


# ==========================================
# 2. ONE-HOT ENCODING (For Nominal Features)
# ==========================================
# Method A: Using Pandas (Easiest for quick analysis)
# drop_first=True avoids the Dummy Variable Trap!
df_pandas_ohe = pd.get_dummies(df, columns=['Color'], drop_first=True)

# Method B: Using Scikit-Learn (Best for Machine Learning Pipelines)
ohe = OneHotEncoder(drop='first', sparse_output=False)
color_encoded = ohe.fit_transform(df[['Color']])
# (You would then concatenate this numpy array back to your main dataframe)


# ==========================================
# 3. LABEL ENCODING (For the Target Variable 'y' ONLY)
# ==========================================
# Never use LabelEncoder for features (X), only for the target (y)
label_encoder = LabelEncoder()
df['Purchased_Encoded'] = label_encoder.fit_transform(df['Purchased'])

# View the final dataset
print("Original Data:\n", df[['Size', 'Color', 'Purchased']])
print("\nPandas get_dummies output:\n", df_pandas_ohe)
print("\nFully Encoded Data (Scikit-Learn approach):\n", df[['Size_Encoded', 'Purchased_Encoded']])
