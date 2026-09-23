# Cross-Validation & Hyperparameter Tuning Cheatsheet

**Type:** Model Selection & Optimization  
**Purpose:** Ensuring your model generalizes to unseen data and finding the absolute best settings for its algorithm.

## Intuition
If you train a model on a dataset and test it on that *exact same* dataset, it might score 100%. But it didn't learn the underlying patterns; it just memorized the answers! This is called **Overfitting**. 

To prevent this, we hold back a piece of the data (the Test Set). But if we tweak our model hundreds of times to get a high score on the Test Set, we accidentally memorize the Test Set too! 

**The Solution:** We need a robust way to evaluate the model (Cross-Validation) and an automated way to twist the model's knobs to find the perfect settings (Hyperparameter Tuning).

## Cross-Validation Techniques

### 1. The Train / Validation / Test Split
Instead of a 2-way split, you do a 3-way split:
*   **Training Set (60-70%):** Used to train the model.
*   **Validation Set (15-20%):** Used to evaluate the model *while* you tweak hyperparameters.
*   **Test Set (15-20%):** Locked in a vault. Used exactly *once* at the very end to report the final real-world performance.

### 2. K-Fold Cross-Validation
What if your single Test Set randomly happened to contain all the easy examples? Your score would be an illusion. 
*   **How it works:** K-Fold splits your entire training dataset into $K$ equal-sized chunks (folds). It trains the model $K$ times. Each time, it holds out a different chunk as the validation set and trains on the remaining chunks.
*   **Result:** You get $K$ different accuracy scores. You take the average of these scores for a much more reliable estimate of how the model will perform in the real world. (Usually, $K = 5$ or $10$).

### 3. Stratified K-Fold
*   A variant of K-Fold used for **Imbalanced Classification**. 
*   It ensures that every single fold contains the exact same ratio of classes as the original dataset (e.g., exactly 90% healthy and 10% sick in every fold).

## Hyperparameter Tuning
Hyperparameters are the "knobs" on an algorithm that you set *before* training (like the depth of a Decision Tree, or the `k` in KNN). Finding the best combination is crucial.

### 1. Grid Search (`GridSearchCV`)
*   **How it works:** You give it a list of values for each hyperparameter. It tests *every single possible combination* of those values using Cross-Validation.
*   **Pros:** Guaranteed to find the best combination from the options you provided.
*   **Cons:** Extremely computationally expensive. If you have 5 parameters with 5 options each, that's $5^5 = 3,125$ models to train!

### 2. Random Search (`RandomizedSearchCV`)
*   **How it works:** You give it ranges for each hyperparameter. It randomly selects a set number of combinations to try (e.g., "Just try 100 random combinations").
*   **Pros:** Vastly faster than Grid Search. Statistically, it usually finds a combination that is 99% as good as Grid Search in a fraction of the time.
*   **Cons:** Might theoretically miss the absolute "perfect" combination.

## The Golden Rule: Prevent Data Leakage!
Data leakage happens when information from your Test Set accidentally leaks into your Training process.
**Rule:** Any preprocessing steps (Scaling, Imputing, Encoding) MUST be done inside the Cross-Validation loop (using Scikit-Learn `Pipeline`), or fit strictly on the training folds. If you scale your entire dataset *before* doing K-Fold, you have leaked data!

## Sample Python Implementation (scikit-learn)

```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# 2. Load and split data (Keeping the Test Set completely separate)
# df = pd.read_csv('data.csv')
# X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# APPROACH A: Simple K-Fold Cross Validation
# ==========================================
model = RandomForestClassifier(random_state=42)
# Run 5-Fold CV on the training data
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print(f"CV Scores: {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean():.4f}")


# ==========================================
# APPROACH B: Grid Search CV
# ==========================================
# Define the exact values to test
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# n_jobs=-1 uses all CPU cores to speed up training
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Grid Search Best Params: {grid_search.best_params_}")
print(f"Grid Search Best CV Score: {grid_search.best_score_:.4f}")


# ==========================================
# APPROACH C: Random Search CV (Faster)
# ==========================================
import numpy as np

# Define distributions or lists to sample from
param_dist = {
    'n_estimators': np.arange(50, 500, 50),
    'max_depth': [None] + list(np.arange(5, 30, 5)),
    'min_samples_split': np.arange(2, 11, 2)
}

# n_iter=20 means it will only try 20 random combinations out of all possibilities
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                   n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print(f"Random Search Best Params: {random_search.best_params_}")
print(f"Random Search Best CV Score: {random_search.best_score_:.4f}")

# ==========================================
# FINAL STEP: Evaluate on the locked-away Test Set
# ==========================================
# Both search objects automatically retrain on the full X_train with the best params
best_model = random_search.best_estimator_
final_accuracy = best_model.score(X_test, y_test)
print(f"Final True Test Accuracy: {final_accuracy:.4f}")
