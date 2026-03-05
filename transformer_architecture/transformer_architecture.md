---
title: "Transformer Architectures"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Transformers have become an essential tool in natural language processing. We have seen the self-attention mechanism in the last lecture, but there is more we need to understand to further proceed. This lecture discusses the core components of Transformers, highlighting encoder and decoder architectures.

## Positional encoding

One unique aspect of the Transformer architecture is that it does not inherently possess a notion of sequence order since it processes input data in parallel. We have already seen this property in our lecture on scaled dot-product attention, which is a core component of transformers. However, understanding the order of tokens in a sequence is critical for capturing contextual relationships. To address this, positions are encoded and combined with the original encoding of each token.

- **Order Awareness:** It provides each token with information about its position in the sequence.
- **Contextual Understanding:** Enables the model to distinguish between sequences like "cat sat" and "sat cat".

The positional encodings are added to the input embeddings at the encoder stage. For a sequence of length $$ B $$ with embedding dimension $$ d $$, the positional encoding $$\mathbf{PE}$$ can be defined as follows:

For each position $$\text{pos}$$, the encoding for each dimension $$i$$ is computed as:

$$
\begin{align}
\mathbf{PE}_{\text{pos}, 2i} &= \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d}}}\right)\\
\mathbf{PE}_{\text{pos}, 2i+1} &= \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d}}}\right)
\end{align}
$$

This choice of sine and cosine functions with varying frequencies enables the model to learn positional relationships effectively. The even and odd indices are handled separately by using sine and cosine, providing a non-linear transformation that helps the model gauge both absolute and relative positions in a sequence.

But why is this positional encoding using sine and cosine functions?

1. **Non-Linearity:** They introduce useful non-linearity and periodicity that help capture sequential patterns.
2. **Smoothness & Differentiability:** These functions provide smooth gradients, aiding in optimization during training.
3. **Ease of Computation:** Easy calculation of positional differences without learned parameters, maintaining consistency across positional dimensions.

Below is just an example implementation of positional encoding using ``torch``and ``numpy``:
```python
import numpy as np
import torch

def get_positional_encoding(max_seq_len, embed_dim):
    positional_encoding = np.zeros((max_seq_len, embed_dim))
    
    for pos in range(max_seq_len):
        for i in range(0, embed_dim, 2):
            positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i)/embed_dim)))
            positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i)/embed_dim)))

    return torch.tensor(positional_encoding, dtype=torch.float32)
```

In this implementation:
- `max_seq_len` is the maximum potential sequence length, providing enough room for different lengths during training.
- `embed_dim` corresponds to the dimensionality of the input embeddings.

By adding these computed positional encodings to the word embeddings, the transformer gains awareness of token positions within sequences, enabling it to make informed predictions based on input structure.

## Encoder and Decoder

A Transformer model can consist of two components:
1. **Encoder** - Maps input sequences into a continuous representation.
2. **Decoder** - Converts this representation into an output sequence, often used in sequence-to-sequence tasks like machine translation.

Both encoder and decoder utilize attention mechanisms to process the input data effectively.


## Encoder Architecture

The encoder is composed of repeated identical layers, each featuring:

1. Multi-Head Self-Attention Layer: Allows each token to attend to other tokens at different subspaces or "heads."
2. Feedforward Layer: Consists of two fully connected layers with a ReLU activation function in between.
3. Add & Norm: A residual connection wraps around each sub-layer, followed by layer normalization.

Let's use some pseudo-code and notation to clarify this further:

For each layer $$i$$ in the encoder with input $$\mathbf{X}_i$$:

$$
\begin{align}
\mathbf{Z}_i &= \text{LayerNorm}(\mathbf{X}_i + \text{MultiHeadAttention}(\mathbf{X}_i))\\
\mathbf{X}_{i+1} &= \text{LayerNorm}(\mathbf{Z}_i + \text{FeedForward}(\mathbf{Z}_i))\\
\end{align}
$$

## Decoder Architecture

The decoder architecture is similar but includes additional components to handle output generation:
1. Masked Multi-Head Self-Attention Layer: Prevents positions from attending to future positions during training.
2. Encoder-Decoder Attention Layer: Enables the model to focus on relevant parts of the input sequence, using outputs from the encoder.
3. Feedforward Layer: see encoder.
4. Add & Norm: Used just like in the encoder for stabilizing and expediting learning.

For each layer $$j$$ in the decoder with input $$\mathbf{Y}_j$$, we have:

$$
\begin{align}
\mathbf{M}_j &= \text{LayerNorm}(\mathbf{Y}_j + \text{MaskedMultiHeadAttention}(\mathbf{Y}_j))\\
\mathbf{N}_j &= \text{LayerNorm}(\mathbf{M}_j + \text{EncoderDecoderAttention}(\mathbf{M}_j, \mathbf{Z}_N))\\
\mathbf{Y}_{j+1} &= \text{LayerNorm}(\mathbf{N}_j + \text{FeedForward}(\mathbf{N}_j))
\end{align}
$$

## Python-Code example: single transformer block

Here is a simplified pseudo-code exemplifying a transformer block using PyTorch:

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        return x
```

## Further Resources

1. **The Annotated Transformer:** An excellent resource that annotates and explains each part of the transformer model. Available [here](https://nlp.seas.harvard.edu/2018/04/03/attention.html).
2. **Diving Deep into Transformer Models:** Detailed articles explaining every aspect of transformers, including applications beyond NLP. Check [this](https://medium.com/@asimsultan2/understanding-transformers-a-simplified-guide-with-easy-to-understand-examples-227fd8d31848).
3. **Understanding the Transformer Model:** Visual guides showing how transformers work under the hood. Find it [here](https://jalammar.github.io/illustrated-transformer/).
