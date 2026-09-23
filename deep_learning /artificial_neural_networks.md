# Artificial Neural Networks (ANN) Cheatsheet

**Type:** Deep Learning (Tabular / Structured Data)  
**Output:** Classification (Discrete labels/probabilities) or Regression (Continuous values)

## Intuition
Artificial Neural Networks (ANNs), specifically **Multi-Layer Perceptrons (MLPs)**, are the foundation of Deep Learning. They are loosely inspired by the biological human brain.

An ANN consists of thousands of interconnected "neurons" arranged in layers:
1.  **Input Layer:** Receives the raw data (features).
2.  **Hidden Layers:** The "brain" of the network where the magic happens. They extract patterns and representations from the data.
3.  **Output Layer:** Delivers the final prediction.

Every connection between neurons has a **Weight** (how important that connection is), and every neuron has a **Bias** (a baseline threshold). Training the network is simply the process of tweaking these millions of weights and biases until the network makes accurate predictions.

## The Core Mechanics

**1. Forward Propagation:**  
Data flows from left to right. Each neuron multiplies the inputs by their weights, adds the bias, and passes the result through an **Activation Function**.
$$z = \sum_{i=1}^{n} (w_i \cdot x_i) + b$$
$$a = \text{ActivationFunction}(z)$$

**2. Activation Functions:**  
Without these, a neural network is just a giant linear regression model, regardless of how many layers it has! Activation functions introduce **non-linearity**.
*   **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$. The default choice for hidden layers. Fast and prevents the vanishing gradient problem.
*   **Sigmoid:** Squashes values between 0 and 1. Used in the output layer for binary classification.
*   **Softmax:** Used in the output layer for multi-class classification. Converts raw scores into probabilities that sum to 1.

**3. Backpropagation & Gradient Descent:**  
Once the network makes a prediction, it calculates the error (Loss). **Backpropagation** applies the Chain Rule of calculus to figure out exactly how much each weight contributed to that error. **Gradient Descent** (usually the `Adam` optimizer) then takes a tiny step to adjust those weights to reduce the error for the next time.

## Key Hyperparameters
*   **Number of Hidden Layers / Neurons:** Defines the "capacity" of the model. Too few = Underfitting. Too many = Overfitting (and slow training).
*   **Learning Rate:** How big of a step the optimizer takes during Gradient Descent. Too large = model bounces around and fails to converge. Too small = training takes forever.
*   **Batch Size:** How many data samples the network looks at before updating its weights (e.g., 32, 64).
*   **Epochs:** How many times the network sees the *entire* training dataset.
*   **Dropout:** A regularization technique where random neurons are "turned off" during training to prevent the network from memorizing the data (overfitting).

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Universal Function Approximators:** Mathematically capable of learning *any* complex, non-linear relationship. | **Black Box:** Almost impossible to explain *why* the network made a specific prediction (lack of interpretability). |
| **High Capacity:** Performance scales incredibly well as you give it more data. | **Data & Compute Hungry:** Requires much more data and processing power (GPUs) than traditional ML models like Random Forest. |
| **Feature Extraction:** Hidden layers automatically learn to extract useful features, reducing the need for manual feature engineering. | **Prone to Overfitting:** Without strict regularization (Dropout, L2), large networks will perfectly memorize the training data and fail on new data. |

## Evaluation Metrics
*   **Classification:** Accuracy, Binary Crossentropy (Loss), Categorical Crossentropy, ROC-AUC.
*   **Regression:** Mean Squared Error (MSE), Mean Absolute Error (MAE).

## Real-World Projects & Use Cases
1.  **Customer Churn Prediction:** Analyzing complex behavioral and demographic data to predict if a user will cancel their subscription.
2.  **Financial Fraud:** Detecting credit card fraud based on thousands of transaction features.
3.  **Pricing Models:** Predicting real estate prices or dynamic ride-sharing fares based on location, time, and historical trends.
4.  **Recommendation Engines:** Serving as the core algorithm for complex, dense collaborative filtering systems.

## Sample Python Implementation (TensorFlow / Keras)

```python
# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 2. Load and Prepare Data (Example: Binary Classification)
# df = pd.read_csv('customer_data.csv')
# X = df.drop('churn', axis=1)
# y = df['churn']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. SCALE THE FEATURES (Mandatory for Neural Networks!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize the ANN
model = Sequential()

# 6. Add Hidden Layers
# Input dimension must match the number of features in X
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3)) # Drops 30% of neurons randomly to prevent overfitting

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))

# 7. Add Output Layer (Binary Classification = 1 neuron, sigmoid activation)
model.add(Dense(units=1, activation='sigmoid'))

# 8. Compile the model
# Adam is the most popular, robust optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 9. Train the model
# batch_size=32 and epochs=50 are common starting points
history = model.fit(
    X_train_scaled, y_train, 
    validation_split=0.2, # Use 20% of training data for validation tuning
    batch_size=32, 
    epochs=50, 
    verbose=1
)

# 10. Evaluate the model on unseen test data
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
