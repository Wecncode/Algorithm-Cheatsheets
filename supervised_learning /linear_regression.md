# Linear Regression - CheatSheet

## Intuition

Linear Regression is one of the simplest and most foundational algorithms in machine learning. Its goal is to model the relationship between a continuous dependent variable (target) and one or more independent variables (features) by fitting a straight line (or a hyperplane in higher dimensions) through the data.

**ELI5:** Imagine you are plotting house sizes on the X-axis and their prices on the Y-axis. Linear regression draws the "best fit" straight line through those points so you can predict the price of a house you haven't seen before.

## The Math

### 1. The Hypothesis (Equation of the Line)

For a dataset with $n$ features, the linear regression model predicts the output $\hat{y}$:

$$
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n
$$

* $\hat{y}$ = Predicted value
* $\beta_0$ = Y-intercept (bias term)
* $\beta_1, \beta_2, \dots, \beta_n$ = Model coefficients (weights)
* $x_1, x_2, \dots, x_n$ = Features

### 2. The Cost Function (Mean Squared Error - MSE)

To find the "best fit" line, we need to minimize the error between our predictions ($\hat{y}$) and the actual values ($y$). We use the Mean Squared Error (MSE):

$$
J(\beta) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

* $N$ = Number of training examples
* $y_i$ = Actual value for the $i$-th example
* $\hat{y}_i$ = Predicted value for the $i$-th example

### 3. Optimization (Gradient Descent)

To minimize the cost function $J(\beta)$, the algorithm iteratively updates the weights using Gradient Descent:

$$
\beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} J(\beta)
$$

* $\alpha$ = Learning rate (controls the step size)

## Assumptions (L.I.N.E.)

For Linear Regression to be statistically reliable, your data should meet these assumptions:

1. **Linearity:** The relationship between features and the target is strictly linear.
2. **Independence:** Observations are independent of each other (no multicollinearity among features).
3. **Normality:** The residuals (errors) are normally distributed.
4. **Equal Variance (Homoscedasticity):** The variance of residual errors should be constant across all levels of the independent variables.

## 🟢 Pros & 🔴 Cons

| **Pros** | **Cons** | 
| ----- | ----- | 
| ✅ Simple to implement and understand. | ❌ Assumes a linear relationship (fails on complex data). | 
| ✅ Highly interpretable (coefficients explain feature importance). | ❌ Highly sensitive to outliers. | 
| ✅ Fast training and prediction times. | ❌ Prone to multicollinearity (features must be independent). | 

## Evaluation Metrics

How do we know if our line is good?

* $R^2$ **(R-Squared):** Represents the proportion of variance in the target variable explained by the model (closer to $1.0$ is better).
* **MAE (Mean Absolute Error):** The average absolute distance between predictions and actual values.
* **RMSE (Root Mean Squared Error):** The square root of MSE. It penalizes larger errors more heavily than MAE.

## Real-World Projects & Use Cases

Linear regression is best used when you need to predict a continuous numeric output based on historical data. Here are common beginner-friendly and industry projects:

* **Real Estate Price Prediction:** Estimating the market value of a house based on features like square footage, number of bedrooms, and zip code.
* **Retail Sales Forecasting:** Predicting future revenue for a store based on historical daily sales, advertising spend, and holiday seasonality.
* **Salary Estimation:** Predicting a candidate's expected salary given their years of experience, education level, and job role.
* **Logistics & ETA Prediction:** Estimating delivery times for shipments based on total distance, historical traffic patterns, and vehicle type.
* **Insurance Premium Calculation:** Assessing the base risk and cost of an insurance premium for a user based on their age, BMI, and medical history.

## Sample Code Snippet (scikit-learn)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data (Example using random data)
X = np.random.rand(100, 1) * 10  # Feature
y = 2.5 * X + np.random.randn(100, 1) * 2  # Target with some noise

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make Predictions
y_pred = model.predict(X_test)

# 5. Evaluate
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"R-Squared: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
```
