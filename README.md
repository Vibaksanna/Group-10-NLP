# 📘 Mini Project 1 – Text Generation using N-gram Language Models

## 🧠 Project Overview

This project is part of the **Natural Language Processing (NLP)** course.  
The objective is to build, compare, and analyze two **4-gram language models** for **text generation**:

- **LM1 – Backoff Language Model** (unsmoothed)
- **LM2 – Interpolation Language Model** (with add-k smoothing)

Both models are trained on a text corpus, evaluated using **perplexity**, and used to generate sample text.

---

## 🎯 Project Objectives

- Understand how **n-gram language models** work
- Handle **data sparsity** using backoff and smoothing techniques
- Compare language models using **perplexity**
- Generate text using classical NLP methods (no deep learning)

---

## 📁 Project Structure

```text
Mini_Project_1/
│
├── data/
│   └── corpus.txt              # Input text corpus
│
├── src/
│   ├── preprocess.py           # Data loading & preprocessing
│   ├── ngram.py                # N-gram frequency construction
│   ├── lm_backoff.py           # Backoff language model (LM1)
│   ├── lm_interpolation.py     # Interpolation language model (LM2)
│   ├── perplexity.py           # Model evaluation
│   └── generate.py             # Text generation
│
├── venv/                       # Python virtual environment (not pushed to git)
├── main.py                     # Main entry point
└── requirements.txt            # Project dependencies

🔁 How the Project Works (Pipeline Flow)

corpus.txt
   ↓
preprocess.py
   ↓
ngram.py
   ↓
lm_backoff.py     lm_interpolation.py
        ↓               ↓
        perplexity.py
              ↓
          generate.py
              ↓
            output
⚠️ Only main.py is executed directly.
All other files are modules called by main.py.

🧩 File-by-File Explanation
data/corpus.txt
Contains raw English text

Acts as training, validation, and test data for the models

main.py
Entry point of the project

Controls the full workflow:

Loads and preprocesses data

Builds n-gram models

Evaluates models using perplexity

Generates sample text

Run this file to execute the project.

preprocess.py
Loads the corpus

Converts text to lowercase

Splits data into:

Training (70%)

Validation (10%)

Test (20%)

Builds a limited vocabulary

Replaces rare words with <UNK>

Why needed:
Raw text must be cleaned and standardized before modeling.

ngram.py
Builds frequency counts for:

Unigrams

Bigrams

Trigrams

4-grams

Why needed:
Language model probabilities are computed from n-gram counts.

lm_backoff.py (LM1)
Implements a 4-gram backoff model

Uses higher-order n-grams when available

Backs off to lower-order n-grams when data is sparse

No smoothing applied

Purpose:
Provides a simple baseline language model.

lm_interpolation.py (LM2)
Combines unigram, bigram, trigram, and 4-gram probabilities

Uses interpolation weights (λ values)

Applies add-k smoothing

Purpose:
Improves robustness and handles unseen events.

perplexity.py
Evaluates language models on test data

Computes perplexity

Why needed:
Perplexity measures how well a model predicts unseen text.
Lower perplexity means better performance.

generate.py
Generates new text using a trained language model

Predicts words sequentially based on probability distributions

Why needed:
Demonstrates the qualitative behavior of each model.

▶️ How to Run the Project
1️⃣ Activate the virtual environment
bash
Copy code
venv\Scripts\activate
2️⃣ Ensure corpus.txt contains text
Plain English text

Several thousand words recommended

3️⃣ Run the project
bash
Copy code
python main.py
📊 Example Output
text
Copy code
Perplexity LM1: 144.49
Perplexity LM2: 280.18

LM1 Generated Text:
natural language processing is commonly used in applications

LM2 Generated Text:
messages language interpret software applications <UNK> learning
📈 Result Interpretation
Lower perplexity = better model

LM1 may outperform LM2 on small or sparse corpora

LM2 may generate less fluent text if smoothing is too strong

These behaviors are expected and discussed in the report

Thanks!


```
