# Deep Learning Cheatsheets

Welcome to the **Deep Learning** section of the Algorithm Cheatsheets repository! 

Deep Learning is a highly specialized subset of Machine Learning that relies on **Artificial Neural Networks** with multiple layers (hence the term "deep"). Inspired by the structure and function of the human brain, these models are exceptionally powerful at finding highly complex, non-linear patterns in unstructured data like images, audio, and raw text.

## Core Concepts

Unlike traditional Machine Learning, which often requires heavy manual "feature engineering" (telling the algorithm exactly what to look for), Deep Learning models perform **Representation Learning**. They automatically figure out which features are most important as data passes through their hidden layers.

The field is largely divided by architecture types based on the data they process:
1.  **Standard Tabular Data:** Artificial Neural Networks (ANNs)
2.  **Spatial Data (Images/Video):** Convolutional Neural Networks (CNNs)
3.  **Sequential Data (Text/Time-Series):** Recurrent Neural Networks (RNNs) & LSTMs
4.  **Advanced Sequences (Modern NLP/Vision):** Transformers

## Directory Contents

Here is what you will find in this directory. Each architecture includes a theoretical Markdown cheatsheet and an accompanying Jupyter Notebook using modern frameworks like `TensorFlow/Keras` or `PyTorch` / `Hugging Face`.

### The Foundations
*   **Artificial Neural Networks (ANN):** [Cheatsheet](./ann.md)
    *   *Best for:* General-purpose deep learning on structured tabular data, or serving as the final output layers for more complex networks.

### Computer Vision
*   **Convolutional Neural Networks (CNN):** [Cheatsheet](./cnn.md)
    *   *Best for:* Image classification, object detection, facial recognition, and medical image analysis.

### Sequence & Temporal Models
*   **Recurrent Neural Networks (RNN & LSTM):** [Cheatsheet](./rnn_lstm.md) 
    *   *Best for:* Processing time-series data, speech recognition, and basic sequential text processing where historical memory is required.

### The Modern Frontier
*   **Transformers & Attention:** [Cheatsheet](./transformers.md) 
    *   *Best for:* State-of-the-art Natural Language Processing (LLMs like GPT/BERT), advanced translation, and increasingly, complex computer vision tasks.

## How to use this section

*   **For Theory:** Read the `.md` files. They contain the mathematical intuition, architectural diagrams, pros and cons, and real-world use cases.
*   **For Practice:** Open the `code/` folder to see how to build, compile, train, and evaluate these complex models using industry-standard libraries. 
*   *Note on Hardware:* Deep learning models are computationally expensive. We highly recommend running these notebooks on Google Colab, Kaggle, or a local machine with a dedicated NVIDIA GPU.

---
*Back to the [Main Repository README](../README.md)*
