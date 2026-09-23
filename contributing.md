# Contributing to Algorithm Cheatsheets

First off, thank you for considering contributing to **Algorithm Cheatsheets**! It's people like you that make the open-source community such a fantastic place to learn, inspire, and create.

Our goal is to build the most accessible, visually clean, and practically useful repository for Machine Learning and Data Science learners. Whether you are fixing a typo, adding a new algorithm, or optimizing the Python code, your help is welcome!

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [What Can I Contribute?](#what-can-i-contribute)
3. [The Standard Cheatsheet Format](#the-standard-cheatsheet-format)
4. [Step-by-Step Contribution Guide](#step-by-step-contribution-guide)
5. [Pull Request Process](#pull-request-process)

---

## Code of Conduct
By participating in this project, you are expected to uphold a welcoming and respectful environment. 
*   Use welcoming and inclusive language.
*   Be respectful of differing viewpoints and experiences.
*   Gracefully accept constructive criticism.
*   Focus on what is best for the community.

---

## What Can I Contribute?

### 1. Add New Algorithms
Missing a crucial algorithm? (e.g., CatBoost, UMAP, Generative Adversarial Networks). Feel free to add it following our standard format.

### 2. Improve Existing Cheatsheets
*   **Clarifications:** Make the "Intuition" sections easier to understand for beginners.
*   **Math Corrections:** Fix any errors in the LaTeX formulas.
*   **Better Code:** Update the Jupyter Notebooks or Python snippets to reflect the latest library versions (e.g., modern `scikit-learn` or `PyTorch` practices).

### 3. Fix Bugs & Typos
Found a broken link, a spelling mistake, or formatting issue? These are perfect first-time contributions! Submit a PR and we'll get it merged quickly.

---

## The Standard Cheatsheet Format

To keep the repository scannable and consistent, every new algorithm `.md` file **must** follow this exact structure. Copy this template for new additions:

```markdown
# [Emoji] [Algorithm Name] Cheatsheet

**Type:** [Supervised / Unsupervised / Deep Learning / Preprocessing]  
**Output:** [Classification / Regression / Clustering / Transformation]

## Intuition
[Explain the algorithm in 2-3 short paragraphs like you are talking to a smart high schooler. Use an analogy if possible.]

## The Math / Core Concepts
[Keep it brief. Use LaTeX for formulas: $$y = mx + b$$]

## Key Hyperparameters
*   **`param_1`:** [What it does and rule of thumb]
*   **`param_2`:** [What it does and rule of thumb]

## Pros and Cons
| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| [Pro 1] | [Con 1] |
| [Pro 2] | [Con 2] |

## Evaluation Metrics / How to Measure
[List the metrics used to evaluate this specific algorithm]

## Real-World Projects & Use Cases
1. [Use case 1]
2. [Use case 2]

## Sample Python Implementation ([Library])
[Include a clean, well-commented, copy-pasteable code snippet]
