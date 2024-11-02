---
title: "Linear Model Optimization"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

We have seen how linear models work and how to define loss functions on the training set
to find the models parameters. However, we still do not know how to minimize loss functions.
In essence, we need to identify the optimal values for the weights and biases that minimize the loss function. 
How do we find optimal values? Sure, let's try the method we learned in school and set the derivative to zero.
The problem here is: our function $$L$$ does not only depend on a single variable, but on multiple ones - the weights and the biases. Therefore, let's learn something about partial deriviatives first.


## Introduction to Partial Derivatives

In multivariable calculus, partial derivatives are used to study functions of several variables. They measure how the function changes as each variable is varied while keeping others constant.

For a function $$ f(x, y) $$, the partial derivative with respect to $$ x $$ is denoted by:

$$
\frac{\partial f}{\partial x} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x, y) - f(x, y)}{\Delta x}
$$

Similarly, the partial derivative with respect to $$ y $$ is:

$$
\frac{\partial f}{\partial y} = \lim_{\Delta y \to 0} \frac{f(x, y + \Delta y) - f(x, y)}{\Delta y}
$$

So basically, it is just about fixing all remaining parameters and computing the derivative for a single
variable. Computing all deriviatives gives us the so called gradient vector:

The gradient vector of a scalar function $$ f(x, y) $$ is denoted by the nabla symbol $$ \nabla $$ and is defined as the vector of its partial derivatives:

$$
\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)
$$

More general, for a function $$ f: \mathbb{R}^n \to \mathbb{R} $$, the gradient is:

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

The gradient vector points in the direction of the greatest rate of increase of the function, which
is an extremely important property we use later on. The length of the gradient vector gives the rate of increase in that direction.

Let's dive into a small example: Consider a simple function $$ f(x, y) = 3x^2 + 2xy + y^2 $$:
The partial derivatives are:
$$
\nabla f = \begin{bmatrix} 6x + 2y \\ 2x + 2y \end{bmatrix}
$$

To find the optimal value of a function, we therefore set the gradient vector to zero.

### Linear Model Optimization

Remember, in linear models, we seek to fit data by finding parameters that minimize a loss function, often the mean squared error (MSE):
$$
L(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - (\mathbf{w}^T \mathbf{x}^{(i)} + b))^2
$$

To find the optimal values for $$ \mathbf{w} $$ and $$ b $$, we utilize an analytical approach. This involves setting the partial derivatives of the loss function with respect to each parameter to zero, solving for the conditions of minimality.

1. **Differentiate**: Compute the partial derivatives of the loss function:

   - With respect to $$ \mathbf{w} $$ (in vector notation):
     $$
     \frac{\partial L}{\partial \mathbf{w}} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - (\mathbf{w}^T \mathbf{x}^{(i)} + b)) \mathbf{x}^{(i)}
     $$

   - With respect to $$ b $$ (single scalar derivative):
     $$
     \frac{\partial L}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - (\mathbf{w}^T \mathbf{x}^{(i)} + b))
     $$

2. **Set Equations to Zero**: To find the minimum, set these derivatives to zero:

   - Weight vector $$ \mathbf{w} $$ equation:
     $$
     \sum_{i=1}^{N} (y_i - (\mathbf{w}^T \mathbf{x}^{(i)} + b)) \mathbf{x}_i = 0
     $$

   - Bias term $$ b $$ equation:
     $$
     \sum_{i=1}^{N} (y_i - (\mathbf{w}^T \mathbf{x}^{(i)} + b)) = 0
     $$

3. **Solve the Equations**: These equations form a system that can be solved simultaneously to find the optimal values of $$ \mathbf{w} $$ and $$ b $$.

### Advanced solution in matrix notation

For simplicity, assume no intercept term, i.e. $$ b = 0 $$.
We can derive that the resulting solution for the optimal value of $$\mathbf{w}$$ with
respect to a minimization of $$L$$ is a normal equation:

  $$
  \mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
  $$

where $$ \mathbf{X} $$ is the matrix of inputs and $$ \mathbf{y} $$ is the vector of outputs.

Can you derive this equation yourself? 
We simply assumed that above matrix is invertible. Can you think about cases where the matrix is singular?
What can we do to handle these cases?
{: .notice--info}


