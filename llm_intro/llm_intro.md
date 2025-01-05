---
title: "Large Language Models Introduction"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Large Language Models (LLMs) have significantly advanced the field of natural language processing, enabling machines to predict and generate human-like text. This lecture will explore how LLMs operate, focusing on their ability to understand context and predict the next sequence of words in a text. 

### What are LLMs?

Large Language Models are deep learning models trained on vast amounts of textual data. Their primary function is to predict the next word in a sequence, effectively generating coherent and contextually relevant text. This predictive capability allows them to perform tasks such as translation, summarization, question-answering, and even creative writing (depending on your expectation :)).

There is a again a great introductory video of 3blue1brown on this topic and I highly recommend watching it as a perfect starter:
<iframe width="420" height="315" src="https://www.youtube.com/embed/LPZh9BOjkQs" frameborder="0"> </iframe>

### Pretraining by Next-Word-Prediction

As we have seen in the video, an important ingredient of LLMs is their ability to predict future text from past text - or an answer (future text) to a context (past text) depending on your view-point.
This is only possible through the following principles:

- **Tokenization**: Text is broken down into smaller units called tokens as we have seen in the previous unit.
- **Embedding**: Each token is transformed into a numerical vector (we also know this already).
- **Sequence Modeling with Transformers**: The Transformer model as a building block uses these vectors to capture dependencies and context over varying lengths using a so called attention mechanisms, i.e. different tokens are weighted and token representations (embeddings) change according to context. 
- **Prediction**: Based on the input sequence, the model predicts the most probable next token.

## Sampling and Temperature in Inference

As we have seen, a classical language model learns the probabilities for each token in a sequence.
You might wonder how to generate text based on probabilities. Simply choosing the most likely token each time can lead to repetitive or overly conservative outputs, lacking creativity and diversity.

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

The [OpenAI Playground](https://platform.openai.com/playground/chat) allows you to play around with different temperature values. Please note that the restriction of the temperature being below 2.0 is set by OpenAI to avoid models creating too much gibberish, however, there is no theoretical reason for this restriction.

There is also a great game called [Semantris](https://research.google.com/semantris/) made by Google to illustrate the power of embedding and similarities.

## Challenges and Considerations

We only briefly touched the tip of the iceberg, however, we can already derive some challenges:

1. There is no fact-checking - since the model just learns likely future text passages from training data, we can not gurantee any facts. Even if the training data is clean and fact-proven, there is no gurantee for the correctness of the predicted text.
2. We again learn from data and all biases that are present in the data will be part of the model as well. Since pre-training is based on heaps of text data gathered from the internet there is no sophisticated curation process possible to clean it up. The only choice left are subsequent training runs with human feedback to *unlearn* biases. However, there is also no gurantee that this works to 100%.
3. There is currently an ongoing debate, whether we already reached an upper limit with pure text data for pre-training ([Talk of Ilya Sutskever, NeurIPS 2024 - starting from 7m](https://www.youtube.com/watch?v=WQQdd6qGxNs))