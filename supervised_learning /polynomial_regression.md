# Polynomial Regression Cheatsheet

**Type:** Supervised Learning (Regression)  
**Output:** Continuous (Real numbers)

## Intuition
Polynomial Regression is an extension of Linear Regression. It is used when the relationship between the independent variable(s) $X$ and the dependent variable $y$ is **non-linear** (e.g., curved). 

Instead of fitting a straight line, it fits a curve to the data points by generating new features that are polynomial combinations of the original features (squaring them, cubing them, etc.). 

*Fun Fact:* Even though it fits a non-linear curve to the data, it is mathematically considered a **Linear Model** because the equation is still linear with respect to the *coefficients* (the $\beta$ weights).

## The Math

**1. Simple Linear Regression Equation (Degree 1):**  
$$y = \beta_0 + \beta_1X$$

**2. Polynomial Regression Equation (Degree 2 - Quadratic):**  
Fits a parabolic curve.  
$$y = \beta_0 + \beta_1X + \beta_2X^2$$

**3. Polynomial Regression Equation (Degree $n$):**  
$$y = \beta_0 + \beta_1X + \beta_2X^2 + \beta_3X^3 + ... + \beta_nX^n$$

## 🎛️ The "Degree" Hyperparameter
The **degree** ($n$) determines how flexible the curve is:
*   **Too Low (e.g., Degree 1):** Underfitting (High Bias). The model is too simple and misses the trend.
*   **Perfect Balance:** Captures the underlying pattern without capturing the noise.
*   **Too High (e.g., Degree 15):** Overfitting (High Variance). The curve hits every single training point perfectly but will fail miserably on unseen data.

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| Can model complex, non-linear relationships. | Extremely prone to **overfitting** if the degree is too high. |
| Provides a better fit than simple linear regression for curved data. | Very sensitive to **outliers** (outliers will heavily distort the curve). |
| Uses the same underlying math as linear regression (easy to compute). | Terrible at **extrapolation** (predictions outside the range of training data will spiral out of control). |

## Evaluation Metrics
Since this is a regression task, we use the same metrics as Linear Regression:
*   **Mean Absolute Error (MAE):** Average of absolute differences.
*   **Mean Squared Error (MSE):** Penalizes larger errors heavily.
*   **Root Mean Squared Error (RMSE):** Interpretable in the same units as the target variable.
*   **R-Squared ($R^2$):** Proportion of variance in $y$ explained by the model (closer to 1.0 is better).

## Real-World Projects & Use Cases
1.  **Epidemiology (Disease Spread):** Modeling the initial infection curve of a virus over time, which rarely grows in a perfectly straight line.
2.  **Economics (Wage vs. Age):** Predicting a person's income based on their age. Income typically rises as people gain experience, peaks in mid-to-late career, and then drops near retirement (forming a quadratic-like curve).
3.  **Physics & Engineering:** Modeling projectile trajectories, braking distances of cars at different speeds, or material degradation over time.
4.  **Agriculture:** Predicting crop yield based on the amount of fertilizer used. Up to a point, fertilizer increases yield, but too much becomes toxic and decreases it.

## Sample Python Implementation (scikit-learn)

To implement Polynomial Regression in `scikit-learn`, we first transform our features using `PolynomialFeatures`, and then feed them into a standard `LinearRegression` model. A `Pipeline` is the cleanest way to do this.

```python
# 1. Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load data (Example)
# df = pd.read_csv('wage_data.csv')
# X = df[['age']]  # Feature must be 2D array
# y = df['income']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the model using a Pipeline
degree = 2 # Change this hyperparameter to adjust the curve flexibility
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Train the model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print(f"Polynomial Degree: {degree}")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-Squared:", r2_score(y_test, y_pred))

# Bonus: To get the actual coefficients from the pipeline:
linear_layer = model.named_steps['linearregression']
print("Intercept (Beta 0):", linear_layer.intercept_)
print("Coefficients:", linear_layer.coef_)
