---
title: "Attention Mechanism"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Attention is a fundamental mechanism in many modern neural network architectures, particularly those dealing with sequence data. This lecture will unpack the details of dot-product attention as presented in the seminal paper of [Vaswani et al, 2017](https://arxiv.org/abs/1706.03762), explaining its operational mechanics and demonstrating its implementation through Python examples.

This lecture will be solely about transforming a sequence of embedding vectors into another sequence of embedding vectors and we will see later on how this can be very useful.

## What is a attention and why it is relevant?

Our language does highly depend on context. For example, if I speak about a model here in my lecture, it is obvious due to the context within the sentence (or paragraph) that I am not referring to model as a profession in the fashion industry but to a model related to machine learning. We have already learned that a suitable representation of text
is a sequence of tokens represented as (embedding) vectors. Embeddings can be learned (or assigned), but the meaning (and therefore the embedding) of each word (token) depends on context - so how can we incorporate
information of the context (all other tokens in the text) in the our token embeddings? 

We somehow need to build up a mechanism that allows the token embeddings to influence (transform) each other, this is exactly where attention comes into play. Attention can be defined as "computing a weighted average of (sequence) elements with the weights dynamically computed based on an input query and elements’ keys". To "refine" the embedding vectors of a sequence based on context, we therefore average over certain value vectors assigned to each element of the sequence with weights computed from the sequence itself. Let's look into the famous "scaled dot-product attention" principle as a concrete example.

## Scaled Dot-Product Attention

Let's assume we converted our text into a sequence of embeddings $$Z = (\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_B)$$
with each being a vector of some embedding dimension.

Dot-Product Attention is a form of self-attention that calculates the importance of each element within a sequence (such as a sequence of tokens) relative to others. By leveraging mathematical operations, it assigns weights to different parts of the input (sequence), focusing on the most relevant pieces of information for further processing.

Given $$Z$$, we compute for each element of the sequence a query, a key and a value vector. This is done 
with fully-connected layers from the inputs to queries, keys, values respectively. 
We therefore assume that we already have a sequence of $$B$$ queries, $$B$$ keys, and $$B$$ values. 
These sequences will be represented as matrices $$\mathbf{Q} \in \mathbb{R}^{B \times d_k}$$,
$$\mathbf{K} \in \mathbb{R}^{B \times d_k}$$, and $$\mathbf{V} \in \mathbb{R}^{B \times d_v}$$ in the following.
Queries and keys have dimension $$d_k$$ and values have dimension $$d_v$$. Yeah, we will have quite some linear algebra again - but don't worry you will get used to it.
The meaning of the matrices is as follows:

1. **Queries (Q):** Represent the items we want to compute attention for.
2. **Keys (K):** Vector representations of each element only used to compute attention scores.
3. **Values (V):** The actual content being processed, weighted by the attention scores.

The output of an attention layer is a weighted sum of the values, where the weights are determined by the interaction between queries and keys. The core operation involves computing the similarity between queries and keys using a dot product, followed by applying a softmax function to obtain attention scores:

$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \cdot \mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} $$

- $$ \mathbf{Q} \cdot \mathbf{K}^T $$ computes the raw attention scores, which it is $$\mathbb{R}^{B \times B}$$ matrix, i.e. an importance score of each element (token) for each other element.
- $$ d_k $$ is again the dimension of the key and query vectors, used to scale the scores.
- The softmax function normalizes all attention scores to ensure they add up to 1, making them interpretable as probabilities.

## Python Implementation Example

Let's look at a simple numpy code example that demonstrates dot-product attention in Python using `numpy`:

```python
import numpy as np

def dot_product_attention(Q, K, V):
    """
    Calculate the dot-product attention.

    Parameters:
    - Q: Query matrix (batch_size, num_queries, depth)
    - K: Key matrix (batch_size, num_keys, depth)
    - V: Value matrix (batch_size, num_values, depth)

    Returns:
    - Output matrix after applying attention
    """
    # Step 1: Calculate raw attention scores
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Step 2: Apply softmax to normalize scores
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Step 3: Weighted sum of the values
    output = np.matmul(attention_weights, V)
    return output
```


## Further resources
1. A great description of the transformer layer with code: [UVA-DLC](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
2. The illustrated transformer, indeed a very good intro: [Illustrated transformer](https://jalammar.github.io/illustrated-transformer/)