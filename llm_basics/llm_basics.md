---
title: "Large Language Models: Fundamentals - Tokens, embeddings, etc."
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

The ``word2vec`` approach offering a method to learn word associations from a large corpus of text by using shallow neural networks. The model optimizes the likelihood of observing target-context word pairs, capturing semantic relationships between words.

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

## Sampling and Temperature in Inference

A classical language model learns the probabilities for each token in a sequence, you might wonder how to generate text based on these probabilities. Simply choosing the most likely token each time can lead to repetitive or overly conservative outputs, lacking creativity and diversity.

Sampling becomes crucial here as it introduces variability by selecting tokens according to their probability distribution rather than just opting for the highest-probability token every time. By incorporating randomness, sampling allows models to generate more varied and interesting text, which is especially useful in creative writing, dialogue generation, or where diverse responses are desirable.

The temperature parameter plays a pivotal role in this process by controlling the level of randomness:
Low Temperature steers the model towards more deterministic and high-probability selections, offering safer and more predictable outputs, high Temperature favors exploration, allowing less probable tokens a higher chance of being chosen, resulting in imaginative and less conventional outputs.

This balance between predictability and creativity helps tailor the behavior of language models across different applications. Here is a Python snippet demonstrating temperature adjustment during sampling:

```python
import numpy as np

def sample_token(probabilities, temperature=1.0):
    adjusted_probs = np.exp(np.log(probabilities) / temperature)
    adjusted_probs /= np.sum(adjusted_probs)  # Normalize the probabilities
    
    return np.random.choice(len(probabilities), p=adjusted_probs)

# Example usage
probabilities = [0.5, 0.3, 0.2]
token_index = sample_token(probabilities, temperature=0.7)
```

## Applications of Large Language Models

LLMs have various applications, including:

1. **Text Generation:** Creating coherent and contextually relevant text, from simple replies to complex articles.
2. **Machine Translation:** Converting text from one language to another with high accuracy.
3. **Question Answering:** Extracting information and providing answers from large datasets.
4. **Sentiment Analysis:** Classifying and interpreting emotions expressed in texts.
5. **Chatbots and Virtual Assistants:** Enhancing user experience through natural and dynamic conversations.