---
title: "Tokens and embeddings, etc."
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Large Language Models (LLMs) have revolutionized the field of natural language processing by enabling machines to understand and generate human language. This lecture will explore how LLMs operate, focusing on essential concepts such as tokenization and the inference process, while also tracing their historical evolution.

## Historical Context Leading to LLMs

Before the advent of LLMs, natural language processing relied heavily on rule-based systems and statistical models. 
Earlier models used N-grams to predict the next word based on previous words. These models were limited by data sparsity and fixed context windows.
The journey towards modern LLMs was a long one (although it might seem like a quick race), involving several very
different ingredients.
An important aspect is the mapping of words to continuous vector spaces, where semantic similarities are captured through proximity in high-dimensional space.

The ``word2vec`` approach offers a method to learn word associations from a large corpus of text by using shallow neural networks. The model optimizes the likelihood of observing target-context word pairs, capturing semantic relationships between words.

## LLM Input Representation: Tokenization

In LLMs, text data is represented through **tokenization**, which breaks down text into manageable units called tokens. Tokens can be words, subwords, or characters depending on the model’s design.

- **Word Tokens:** Entire words are treated as individual tokens.
  - Example: "Machine learning" becomes ["Machine", "learning"].
  
- **Subword Tokens:** Words are divided into meaningful subunits.
  - Example: The word "unbelievable" might be tokenized as ["un", "believ", "able"].
  
- **Character Tokens:** Each character acts as a separate token.
  - Example: "AI" becomes ["A", "I"].

Tokens enable the model to process input data efficiently by converting it into a structured format suitable for computation. There is a current line of research that investigates models that skip tokenization and directly 
operate on byte level.

## From tokens to vectors: token embeddings

A token id is just an integer and carries no meaning by itself. Therefore, each token id is mapped to a vector, the **token embedding**, by a simple lookup table
with one row per token in the vocabulary. In ``torch``, this table is the ``nn.Embedding`` layer:

```python
import torch
import torch.nn as nn

vocab_size = 10000      # number of different tokens
embed_dim = 64          # dimension of each token embedding

embedding = nn.Embedding(vocab_size, embed_dim)

# a "sentence" given as a sequence of 5 token ids
token_ids = torch.tensor([42, 17, 9000, 3, 17])
vectors = embedding(token_ids)
print(vectors.shape)    # torch.Size([5, 64])
```

The rows of this table are parameters of the model and are learned jointly with all other parameters during training - the model itself
decides which vectors are useful for predicting the next token. The result has the same flavor as the learned embeddings we
have seen in [our lecture on feature vectors](/modules/featurevectors/featurevectors.md): tokens with a similar meaning end up with similar vectors.



## Applications of Large Language Models

LLMs have various applications, including:

1. **Text Generation:** Creating coherent and contextually relevant text, from simple replies to complex articles.
2. **Machine Translation:** Converting text from one language to another with high accuracy.
3. **Question Answering:** Extracting information and providing answers from large datasets.
4. **Sentiment Analysis:** Classifying and interpreting emotions expressed in texts.
5. **Chatbots and Virtual Assistants:** Enhancing user experience through natural and dynamic conversations.
