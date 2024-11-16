---
title: "Pytorch essentials"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

PyTorch is a powerful deep learning library popular for its flexibility and ease of use. It allows dynamic computation, making it ideal for research and development. Alternatives are tensorflow or similar libraries.

Key Features are:
- **Dynamic Computation Graphs**: Provides flexibility in building complex architectures.
- **GPU Acceleration**: Efficient computation on GPUs using CUDA (for NVIDIA) or Metal Performance Shaders (for Apple GPUs).
- **Rich API**: Supports neural networks, optimization algorithms, etc.

## Tensors

The most fundamental concept in pytorch is a tensor.
Tensors are n-dimensional arrays that run on CPUs or GPUs for hardware acceleration.
Their API is very similar and to a large extent compatible to the one that we have
seen for numpy arrays.

Creating tensors:
```python
import torch

# Create a tensor
tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Random tensor
tensor_b = torch.rand((3, 3))

# Zero tensor
tensor_c = torch.zeros_like(tensor_a)

print("Tensor A:\n", tensor_a)
print("Tensor B:\n", tensor_b)
print("Tensor C:\n", tensor_c)
```

Basic operations:
```python
# Addition
result_add = tensor_a + tensor_b

# Matrix multiplication
result_mul = torch.matmul(tensor_a, tensor_b[:2])

# Element-wise multiplication
result_elem_mul = tensor_a * tensor_a

print("Addition Result:\n", result_add)
print("Matrix Multiplication Result:\n", result_mul)
print("Element-wise Multiplication Result:\n", result_elem_mul)
```

### GPU handling

PyTorch integrates seamlessly with GPU computing for accelerated operations.
In the majority of cases, you will use an NVIDIA GPU, which uses the CUDA
framework to communicate with the card:

```python
# Check if CUDA is available
if torch.cuda.is_available():
    tensor_gpu = tensor_a.to('cuda')
    print("Tensor moved to GPU")
else:
    print("CUDA is not available. Tensor stays on CPU.")
```

MacBooks generally do not support CUDA as they come with integrated GPUs from Apple or Intel. Available from macOS 12.3, GPU computation is supported on MacBooks through MPS:

```python
# Use MPS if available
if torch.backends.mps.is_available():
    tensor_mps = tensor_a.to('mps')
    print("Tensor moved to GPU via MPS")
else:
    print("MPS is not available. Tensor stays on CPU.")
```

## Gradients and Computation Graphs

One of the main features of pytorch is to automatically calculate gradients such
from a given computation defined by tensor operations.
To illustrate this, we often use the concept of a computation graph, where
nodes represent operations (or initial variables) and the directed acyclic graph shows the
inputs and outputs of the operations.
In PyTorch, the computation graph is built dynamically as operations are performed.

Let's directly dive into an example. We first define a simple tensor $$x$$ (one dimension only) and then compute $$y$$ depending on $$x$$. 
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 3 * x ** 2 + 5
```

The gradient $$\nabla_x y$$ (or $$\frac{\mathrm{d}y}{\mathrm{d}y}$$ in our simple case) can then be automatically computed by:

```python
# Backpropagate to populate gradients
y.backward()

# Access the gradient
print("Gradient of y with respect to x:", x.grad)
```

In this example, `x` has `requires_grad=True`, indicating it should be tracked by the pytorch's automatic differentiation system and some space is reserved for the gradient.



You can perform complex operations and still retrieve gradients efficiently.

```python
# New tensors with gradient requirement
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# A more complex function
z = a ** 2 * b + b ** 3

# Backward pass and calculate each variable's gradient
z.backward()

print(f"Gradient of z wrt a: {a.grad}")  # dz/da
print(f"Gradient of z wrt b: {b.grad}")  # dz/db
```

For $$z = a^2 \cdot b + b^3$$, we have:

$$
\begin{align}
\frac{dz}{da} &= 2a \cdot b = 2 \cdot 2 \cdot 3 = 12\\
\frac{dz}{db} &= a^2 + 3b^2 = 4 + 27 = 31
\end{align}
$$


To improve performance and reduce memory usage during evaluation, you can also disable gradient calculations:

```python
# Evaluation mode without gradient tracking
with torch.no_grad():
    eval_result = a * b + b**2
    print("Evaluation Result (no grad):", eval_result)
```

## Dynamic computation graphs

PyTorch's dynamic computation graph is **built on-the-fly**. This design offers numerous advantages, particularly in scenarios involving varying execution paths or iterative computations that change over time.

```python
import torch

def dynamic_graph_example(input_tensor):
    # Sample dynamic condition
    if input_tensor.sum() > 0:
        return input_tensor * 2
    else:
        return input_tensor - 2

tensor = torch.tensor([1.5, -3.2], requires_grad=True)
result = dynamic_graph_example(tensor)

# The graph is created dynamically based on the input tensor
print("Result:", result)
```

In this example, the operation within the function `dynamic_graph_example` depends on the input tensor values, illustrating runtime graph construction adaptability.

Another example are Recurrent Neural Networks (RNNs):

```python
def dynamic_rnn(x, hidden_size=10):
    h = torch.zeros(hidden_size, dtype=torch.float32)
    for t in range(x.size(0)):
        h = torch.tanh(x[t] @ h)
    return h

# Sequence of inputs
sequence_input = torch.randn((5, 10), requires_grad=True)
output_h = dynamic_rnn(sequence_input)

# Compute gradients for entire sequence
output_h.backward(torch.ones_like(output_h))
```

In this RNN example, each time step adapts operations based on the sequence, showcasing the power of dynamic computation graphs.