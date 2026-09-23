# Transformers & Attention Cheatsheet

**Type:** Deep Learning (NLP, Sequence-to-Sequence, Computer Vision)  
**Output:** Generative Text, Sequence Classification, Embeddings, Translation

## Intuition
Before 2017, Recurrent Neural Networks (RNNs) and LSTMs were the standard for text. But they had a fatal flaw: they had to read data sequentially (word by word), making them painfully slow to train and bad at remembering words from the beginning of long paragraphs.

In 2017, Google published the paper *"Attention Is All You Need"*, introducing the **Transformer**. Transformers process the *entire* sequence at once (massively parallelizable) and use a mechanism called **Self-Attention** to figure out which words in a sentence are most strongly related to each other, regardless of how far apart they are.

## The Core Concepts

**1. Self-Attention (Queries, Keys, Values):**  
Think of it like a database search. 
*   **Query (Q):** What the current word is looking for.
*   **Key (K):** What other words in the sentence offer.
*   **Value (V):** The actual meaning/content of the word.
The model takes the dot product of Q and K to find a "score" of how relevant two words are to each other, then multiplies it by V.
$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**2. Multi-Head Attention:**  
Instead of doing Self-Attention once, the Transformer does it multiple times in parallel (multiple "heads"). One head might learn to pay attention to grammar, another to gender pronouns, and another to historical facts.

**3. Positional Encoding:**  
Because Transformers read everything at once, they have no concept of word order. The sentence "Dog bites man" looks the same as "Man bites dog." To fix this, we inject mathematical patterns (usually sine/cosine waves) into the input embeddings so the model knows the *position* of every word.

## Transformer Architectures
*   **Encoder-Only (BERT):** Great at understanding context and extracting information. Used for text classification, sentiment analysis, and search engines.
*   **Decoder-Only (GPT, Llama):** Great at generating text. Predicts the next word in a sequence based on all previous words.
*   **Encoder-Decoder (T5, BART):** The original architecture. Great for sequence-to-sequence tasks like language translation or summarizing articles.

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Massively Parallel:** Can be trained across thousands of GPUs simultaneously, unlike RNNs. | **Quadratic Complexity:** Memory and compute scale quadratically ($O(n^2)$) with sequence length. A 2x longer document requires 4x the memory. |
| **Long-Range Context:** Attention connects distant words directly, entirely eliminating the Vanishing Gradient problem for long documents. | **Data Hungry:** Requires incredibly massive datasets (billions/trillions of tokens) to train from scratch. |
| **State-of-the-Art:** The undisputed king of NLP, and rapidly taking over Computer Vision (Vision Transformers - ViT). | **Extremely Expensive:** Training a large language model (LLM) costs millions of dollars in compute. |

## Evaluation Metrics
*   **Text Generation / Translation:** BLEU Score, ROUGE Score, METEOR.
*   **Language Modeling:** Perplexity (How surprised the model is by the actual next word. Lower is better).
*   **Classification:** Accuracy, F1-Score (GLUE / SuperGLUE benchmarks).

## Real-World Projects & Use Cases
1.  **Large Language Models (LLMs):** ChatGPT, Claude, and Gemini generating human-like text, writing code, and answering complex queries.
2.  **Advanced Search:** Google uses BERT to understand the *intent* behind search queries rather than just matching keywords.
3.  **Code Autocomplete:** GitHub Copilot using OpenAI models to suggest entire blocks of code in real-time.
4.  **Computer Vision:** Vision Transformers (ViT) outperforming traditional CNNs in image classification on massive datasets.

## Python Implementation (Hugging Face `transformers`)

*Writing a Transformer from scratch in PyTorch is very complex. The industry standard is to use pre-trained models from the Hugging Face library. (Run `pip install transformers torch`)*

```python
# 1. Import pipeline from Hugging Face
from transformers import pipeline

# 2. Example 1: Sentiment Analysis using a pre-trained Encoder (BERT-like)
print("--- Sentiment Analysis ---")
classifier = pipeline("sentiment-analysis")
result = classifier("I absolutely loved the new sci-fi movie, the visuals were stunning!")
print(result) 
# Output: [{'label': 'POSITIVE', 'score': 0.999}]

# 3. Example 2: Text Generation using a pre-trained Decoder (GPT-like)
# Note: In a real project, you would use a larger model like GPT-2 or Llama
print("\n--- Text Generation ---")
generator = pipeline("text-generation", model="gpt2")
prompt = "The future of artificial intelligence is"

# Generate up to 30 words
generated_text = generator(prompt, max_length=30, num_return_sequences=1)
print(generated_text[0]['generated_text'])
