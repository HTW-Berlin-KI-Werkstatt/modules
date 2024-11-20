---
title: "Backpropagation"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Basically we have everything we need to jump to back-propagation: back-propagation is simply
 figuring out how to compute all gradients of a loss function automatically and efficiently, such
 that we can use it for gradient descent optimization. We have already seen how back-propagation can be used in pytorch - remember the ``.backwards()`` command? 

## Backpropagation - basic principle

Let's assume that we have a very simple neural network model with a single hidden layer:

$$
\begin{align}
f(\mathbf{x}; \mathbf{w}) &= f_2(f_1(\mathbf{x}; w_1, w_2 ); w_3, w_4)\\
f_1(\mathbf{x}; w_1, w_2 ) &= A(w_1 x_1 + w_2 x_2)\\
f_2(z; w_3, w_4 ) &= A(w_3 z + w_4)
\end{align}
$$

with parameters $$\mathbf{w} = (w_0, w_1, w_2, w_3)$$ and $$A$$ being the sigmoid function.
How can we find good parameters $$\mathbf{w}$$? We also focus only on a single example $$\mathbf{x}=(1,1)$$ with label $$y=2$$. The model output $$f(\mathbf{x})$$ needs to be close to $$y$$.

Let's do one *forward pass*, i.e. we compute $f$ for some initially chosen parameters $\mathbf{w}$. We also store all intermediate values, such as the output of $f_1(\cdot) = z$. 
Let's assume the label $$y_i$$ is $2$ but the model output $f(\cdot)$ is $0.1$ far from being ideal. What can be done in the last set of weights before the output?

1. We could increase the bias term $$w_4$$,
2. increase the coefficient $$w_3$$ of $$z$$,
3. or increase $z$ the output of $f_1$.

Increasing $$w_3$$ leads to a change of $$f_2$$ which is proportional to $$z$$ - the impact of a change in $$w_3$$ is obviously larger when $$z$$ is large as well. 
But how can we change $$z$$? We have to increase the parameters $$w_1$$ or $$w_2$$ ($$x_1$$ and $$x_2$$ are positive) and the impact of the change depends on $$x_1$$ and $$x_2$$. 

This step-wise explanation shows:
1. We first considered the last set of weights (last layer) and then moved towards the input - this is exactly where the name **back**propagation comes from,
2. There are various options how to change the output of the function and if the function would have multiple outputs they might be in conflict with each other.

Again, nothing can beat the great videos of 3Blue1Brown and the following video
gives a great overview of backpropagation without going into calculus too much:
<iframe width="420" height="315" src="https://www.youtube.com/embed/Ilg3gGewQ5U" frameborder="0"> </iframe>

## Backpropagation = computing derivatives efficiently

Just watch the following video :-P

<iframe width="420" height="315" src="https://www.youtube.com/embed/tIeHLnjs5U8" frameborder="0"> </iframe>
