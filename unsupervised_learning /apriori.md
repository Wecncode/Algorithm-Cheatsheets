# Apriori Algorithm Cheatsheet

**Type:** Unsupervised Learning (Association Rule Mining)  
**Output:** A set of "If-Then" rules connecting items (e.g., If {Bread, Butter} $\rightarrow$ Then {Milk})

## Intuition
Apriori is the foundational algorithm for **Market Basket Analysis**. It is used to discover hidden patterns, correlations, and co-occurrences in large transactional datasets.

Instead of predicting a target variable, it asks: *"Which items are frequently bought together?"* 

To avoid calculating every single possible combination of items (which would take centuries for a supermarket), it uses the **Apriori Property** to drastically reduce the search space.

## The Apriori Property (Pruning)
**The Golden Rule:** *All non-empty subsets of a frequent itemset must also be frequent.*
*   **Meaning:** If people rarely buy {Beer} by itself, they will also rarely buy {Beer, Diapers} together. 
*   **Action:** As soon as the algorithm finds an item (or small group of items) that is infrequent, it instantly deletes *all* larger combinations containing that item, saving immense computational time.

## The Math (Key Metrics)
To evaluate how "strong" or "relevant" a rule like $A \rightarrow B$ is, Apriori relies on three core metrics:

**1. Support:**  
How popular is the itemset? (Frequency of items A and B appearing together in all transactions).
$$Support(A \rightarrow B) = \frac{\text{Transactions containing } A \text{ and } B}{\text{Total Transactions}}$$

**2. Confidence:**  
If a customer buys A, how likely are they to buy B? (Directional conditional probability).
$$Confidence(A \rightarrow B) = \frac{Support(A \cup B)}{Support(A)}$$

**3. Lift:**  
How much does buying A *increase* the probability of buying B, compared to if B was just bought randomly?
$$Lift(A \rightarrow B) = \frac{Confidence(A \rightarrow B)}{Support(B)}$$
*   *Lift = 1:* A and B are completely independent (no relationship).
*   *Lift > 1:* A and B are positively correlated (good rule!).
*   *Lift < 1:* A and B are negatively correlated (substitutes).

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Highly Interpretable:** Outputs simple, human-readable "If-Then" rules. | **Computationally Expensive:** Even with pruning, it struggles with massive datasets with thousands of unique items. |
| **No Preprocessing Heavy-Lifting:** Doesn't require scaling, normalizing, or handling continuous math. | **Spurious Rules:** Can generate thousands of meaningless rules if minimum thresholds are set too low. |
| **Easily Actionable:** Direct applications in business strategy (store layouts, cross-selling). | **Requires Discretization:** Cannot handle continuous numerical data natively (must be converted to categories/bins). |

## Real-World Projects & Use Cases
1.  **E-Commerce Recommendations:** "Customers who bought this item also bought..." (Amazon).
2.  **Retail Store Layout:** Placing complementary items next to each other in physical aisles (e.g., Chips and Salsa) to maximize spontaneous purchases.
3.  **Healthcare Symptom Checker:** Discovering co-occurring symptoms to aid in diagnosing complex diseases.
4.  **Content Consumption:** Suggesting Spotify songs or Netflix movies based on sequences of media frequently consumed together.

## Sample Python Implementation (mlxtend)

*Note: `scikit-learn` does NOT have a built-in Apriori algorithm. The industry standard for this in Python is `mlxtend`. You may need to run `pip install mlxtend`.*

```python
# 1. Import libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 2. Load and Prepare Data
# Apriori requires a One-Hot Encoded boolean dataframe
# Rows = Transactions, Columns = Items, Values = True/False
data = {
    'Bread':  [True, True, False, True, True],
    'Butter': [True, True, False, True, False],
    'Milk':   [True, False, True, True, False],
    'Apples': [False, True, True, False, True]
}
df = pd.DataFrame(data)

# 3. Find Frequent Itemsets (Set a minimum Support threshold)
# min_support=0.4 means the itemset must appear in at least 40% of transactions
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:\n", frequent_itemsets)

# 4. Generate Association Rules (Set a metric and minimum threshold)
# We want rules with a minimum Confidence of 70%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# 5. Sort and View Top Rules
# Sorting by Lift to find the strongest correlations
rules = rules.sort_values(by="lift", ascending=False)

print("\nTop Association Rules:\n")
# Displaying only the most relevant columns for readability
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
