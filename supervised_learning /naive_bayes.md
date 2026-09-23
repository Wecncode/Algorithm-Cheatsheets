# Naive Bayes Cheatsheet

**Type:** Supervised Learning (Classification)  
**Output:** Discrete (Class label based on highest probability)

## Intuition
Naive Bayes is a probabilistic classifier based on **Bayes' Theorem**. It calculates the probability of each class given a set of features and predicts the class with the highest probability.

It is called **"Naive"** because it makes a massive assumption: **all features are completely independent of each other** given the class label. In the real world, this is almost never true (e.g., in an email, the word "free" and the word "money" are definitely related), but surprisingly, the algorithm still works incredibly well in practice.

## The Math (Bayes' Theorem)
Bayes' Theorem finds the probability of an event occurring given the probability of another event that has already occurred.

**1. The Base Formula:**
$$P(A\vert{}B) = \frac{P(B\vert{}A) \cdot P(A)}{P(B)}$$
*   **$P(A\vert{}B)$ (Posterior):** Probability of class A given feature B.
*   **$P(B\vert{}A)$ (Likelihood):** Probability of feature B given class A.
*   **$P(A)$ (Prior):** Initial probability of class A.
*   **$P(B)$ (Evidence):** Total probability of feature B.

**2. Naive Bayes Formula (Multiple Features):**
Because of the "naive" independence assumption, we can just multiply the individual likelihoods together:
$$P(y\vert{}x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i\vert{}y)$$
*(The model drops the denominator $P(B)$ because it is constant for all classes and we only care about which class has the highest score).*

## Types of Naive Bayes Classifiers
1.  **Gaussian Naive Bayes:** Used when features are continuous (real numbers). Assumes the features follow a normal (Gaussian) distribution.
2.  **Multinomial Naive Bayes:** Used for discrete counts. Extremely popular for text classification (e.g., word counts or term frequencies).
3.  **Bernoulli Naive Bayes:** Used for binary/boolean features (e.g., whether a specific word appears in a document or not: 0 or 1).

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Blazing Fast:** Extremely quick to train and predict; scales well to massive datasets. | **The "Naive" Assumption:** Assumes features are independent, which hurts accuracy if features are highly correlated. |
| **High Dimensions:** Performs exceptionally well with high-dimensional data (like text). | **Zero-Frequency Problem:** If a categorical feature value wasn't seen in training, it assigns a 0 probability, wiping out all other calculations (solved by Laplace Smoothing). |
| **Small Data:** Requires less training data than complex models like Logistic Regression. | **Bad Estimator:** It is a good classifier, but the actual probability outputs shouldn't be taken too literally. |

## Evaluation Metrics
*   **Accuracy:** Good for balanced datasets.
*   **Precision & Recall:** Crucial for text classification like spam (where False Positives—sending a real email to spam—are very bad).
*   **F1-Score:** Harmonic mean of Precision and Recall.

## Real-World Projects & Use Cases
1.  **Spam Filtering:** Analyzing the frequency of words like "lottery", "click", and "prince" to classify emails as Spam or Not Spam.
2.  **Sentiment Analysis:** Categorizing movie reviews or tweets as Positive, Negative, or Neutral based on the vocabulary used.
3.  **Document Categorization:** Automatically sorting news articles into topics like "Sports", "Politics", or "Technology".
4.  **Medical Diagnosis:** Predicting a disease based on the presence or absence of independent symptoms.

## Sample Python Implementation (Scikit-learn)

Here is an example using `GaussianNB` for continuous data. (Swap for `MultinomialNB` if you are doing text classification with a `CountVectorizer`).

```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 2. Load data (Example: Predicting if a user clicks an ad based on age and salary)
# df = pd.read_csv('social_network_ads.csv')
# X = df[['Age', 'EstimatedSalary']]
# y = df['Purchased']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Initialize and train the model
# GaussianNB doesn't require hyperparameter tuning!
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
