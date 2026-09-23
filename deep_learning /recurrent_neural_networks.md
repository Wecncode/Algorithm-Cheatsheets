# Recurrent Neural Networks (RNN & LSTM) Cheatsheet

**Type:** Deep Learning (Sequential Data)  
**Output:** Classification (e.g., Sentiment), Sequence Generation (e.g., Translation), or Continuous (e.g., Time Series Forecasting)

## Intuition
Traditional Neural Networks and CNNs assume that all inputs are completely independent of each other. But what if you are reading a sentence? You can't understand the word "apple" without knowing if the previous word was "eating an" or "Apple computer". 

**Recurrent Neural Networks (RNNs)** are designed for sequential data. They have an internal "loop" or memory. As they process an input at time step $t$, they also pass their internal "hidden state" forward to the next time step $t+1$. This allows the network to remember the past to help it predict the future.

## The Core Concepts

**1. The Hidden State ($h_t$):**  
The "memory" of the network. It is calculated based on the current input ($x_t$) and the previous hidden state ($h_{t-1}$).
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

**2. Backpropagation Through Time (BPTT):**  
To train an RNN, the network is "unrolled" for all the time steps, and the errors are backpropagated from the last time step all the way back to the first. 

**3. The Vanishing Gradient Problem:**  
Because gradients (errors) are repeatedly multiplied by weights as they travel backward through time, if those weights are small, the gradient shrinks exponentially. The network "forgets" long-term dependencies (e.g., it remembers the last 3 words, but forgets the beginning of the paragraph). 

## The Saviors: LSTM & GRU
To solve the Vanishing Gradient problem, researchers invented advanced RNN cells that can hold onto information for much longer.

**1. Long Short-Term Memory (LSTM):**  
Instead of just a hidden state, LSTMs have a **Cell State** (a long-term memory conveyor belt). They use three mathematical "gates" to carefully control what information is added or removed from the cell state:
*   *Forget Gate:* Decides what old information to throw away.
*   *Input Gate:* Decides what new information to add.
*   *Output Gate:* Decides what to output based on the cell state.

**2. Gated Recurrent Unit (GRU):**  
A newer, streamlined version of the LSTM. It combines the forget and input gates into a single "Update Gate." It is faster to train and uses less memory, often achieving similar performance to LSTMs.

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Handles Sequential Data:** The only traditional architecture natively built for time series and text. | **Extremely Slow Training:** Because time step $t+1$ requires the output of step $t$, it cannot be parallelized well (unlike CNNs). |
| **Variable Length Inputs:** Can process sequences of any length (e.g., sentences of 5 words or 50 words). | **Vanishing Gradient (Standard RNN):** Standard RNNs are virtually useless for long sequences. |
| **Contextual Memory:** Captures temporal dynamics and historical context perfectly. | **Resource Intensive:** LSTMs and GRUs require heavy computational power and memory due to multiple gates per cell. |

## Evaluation Metrics
*   **NLP/Text:** BLEU Score, Perplexity, ROUGE Score.
*   **Classification:** Accuracy, F1-Score, Confusion Matrix.
*   **Time Series:** RMSE, MAE.

## Real-World Projects & Use Cases
1.  **Natural Language Processing (NLP):** Machine translation (Google Translate's earlier versions), sentiment analysis, and text generation.
2.  **Speech Recognition:** Converting audio waveforms (time series data) into text (e.g., Siri or Alexa).
3.  **Time Series Forecasting:** Predicting tomorrow's stock prices or next week's weather based on historical trends.
4.  **Music Generation:** Learning the patterns of classical music to generate brand new, original compositions.

## Sample Python Implementation (TensorFlow / Keras)

Here is a classic Many-to-One architecture using an LSTM for sentiment analysis (predicting if a movie review is positive or negative).

```python
# 1. Import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Assume data is padded sequences of integers (words mapped to IDs)
# input_length = 100 (max 100 words per review)
# vocab_size = 10000 (10k most common words)

# 2. Initialize the model
model = Sequential()

# 3. Add an Embedding Layer 
# Turns word IDs into dense vectors of fixed size
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))

# 4. Add the LSTM Layer
# return_sequences=False because we only want a final summary output, not an output at every time step
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.5)) # Prevent overfitting

# 5. Add Fully Connected Output Layer
# Binary classification (Positive vs Negative)
model.add(Dense(units=1, activation='sigmoid'))

# 6. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. View the summary
model.summary()

# (Training would use model.fit(X_train, y_train, epochs=5, batch_size=64))
