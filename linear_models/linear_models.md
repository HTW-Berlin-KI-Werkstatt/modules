---
title: "Linear Models and loss functions"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Linear models are foundational techniques in machine learning, widely used for both regression and classification tasks. They are essential due to their simplicity, (pseudo-)interpretability, and efficiency.
For regression tasks, they assume a linear relationship between inputs and outputs, making them easy to understand and implement. For classification tasks, logistic regression provides a powerful method to predict class probabilities and define decision boundaries.

By understanding and leveraging linear models, you gain a solid foundation that is essential when exploring more complex machine learning algorithms.

A linear model aims to predict the target variable $$y$$ as a linear combination of input features $$\mathbf{x} = (x_1, \ldots, x_D)^T$$ as follows:
$$
f(\mathbf{x}; \mathbf{w}, b) = b + w_1 x_1 + w_2 x_2 + \cdots + w_D x_D + \epsilon
$$

Let's recall all the notation in the equation again:
- $$y$$ is our target variable (as usual).
- $$x_1, x_2, \ldots, x_D$$ are the independent input variables (our features).
- $$b$$ is the intercept (or bias, or offset).
- $$w_1, w_2, \ldots, w_D$$ are the coefficients (weights).

Our machine learning model simply consists of the coefficients $$\mathbf{w} = (w_0, w_1, \ldots, w_D)^T$$ and $$b$$. Training therefore is finding again the parameters of such a model. But how do we do this?
First of all, we need to think about how to measure *suitability* of model parameters - how can we determine whether a parameter set fits to a training set.
Indeed, this was a question already asked by Gauß and fellows once upon a time and the main idea is to measure the squared error of the predicted value $$f(\mathbf{x}^{(i)}; \mathbf{w}, b)$$ and $y_i$. 
Please note that we are now using the notation $$\mathbf{x}^{(i)} = (x^{(i)}_1, \ldots, x^{(i)}_D)$$ to refer to training example $i$.
Summing all the squared errors gives us the final objective function (the one that we like to minimize):

$$
L(\mathbf{w}, b) = \sum_{i=1}^{n} (y_i - f(\mathbf{x}^{(i)}; \mathbf{w}, b))^2
$$

Why are we not using a simple difference between the predicted and the ground-truth value? What is the effect of the square-operation? Any idea how to minimize this beast?
{: .notice--info}

The functions like $$L$$ are also known as **loss functions** in machine learning, they
express the error made on a training set by a model. Reducing the error by changing the parameters
of the model appropiately is the goal of the training. 

#### Why Linear Models?

Linear models make sense for regression because they assume a straight-line relationship between the dependent and independent variables. This assumption simplifies the computation and makes the model easy to interpret. They work well when:

- The relationship between the features and the target is approximately linear.
- You need an easily interpretable model.
- Speed and performance are critical, and complex models aren't necessary.

Using a linear combination of features to predict the target variable ensures that the model captures the additive effects of each feature. Even if the data is not perfectly linear, linear models provide a good baseline and often serve as a benchmark for more complex methods.

### Real example for linear regression

Here is a simple implementation using `scikit-learn` with linear regression:

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=10)

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plotting
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Linear regression line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
```


### Application to Classification: Logistic Regression

In binary classification tasks, **Logistic Regression** is a popular choice when employing linear models. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability of an input belonging to a particular class.

At the core of logistic regression is the logistic function (or sigmoid function), defined as:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
where $$z = \mathbf{w}^T \mathbf{x} + b$$. This function maps any real-valued number into the range [0, 1], making it suitable for predicting probabilities.

The model outputs a probability score that represents the likelihood of the data point belonging to a certain class. By applying a threshold (commonly 0.5), we can assign a binary label:
If $$\sigma(z) \geq 0.5$$, class 1 is predicted, otherwise class 0 is predicted.

To train a logistic regression model, we minimize the cross-entropy loss function:

$$
L(\mathbf{w}, b) = - \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

where $$\hat{y}_i = \sigma(\mathbf{w} \cdot \mathbf{x}^{(i)} + b)$$ is the predicted probability for sample $$i$$, and $$y_i$$ is the actual label.

### Extending to Multiclass: Softmax Regression

For multiclass classification tasks, where there are more than two classes, we use the **Softmax Regression** (also known as Multinomial Logistic Regression).

The softmax function generalizes the logistic function to $$K$$ multiple classes. For a given input $$\mathbf{x}$$ and a set of weights $$\mathbf{W} \in \mathbb{R}^{D \times K}$$ and biases $$\mathbf{b} \in \mathbb{R}^{D}$$, the probability that the input belongs to class $$k$$ is given by:
$$
P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$
where $$z_k = \mathbf{w}_k^T \mathbf{x} + b_k$$, $$\mathbf{w}_k$$ is the $$k$$th column of $$\mathbf{W}$$ 
and $$K$$ is the total number of classes.

Similar to binary logistic regression, the multivariate scenario uses categorical cross-entropy loss to optimize the parameters:

$$
\begin{align}
L(\mathbf{W}, \mathbf{b}) &= - \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})\\
                          &= - \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log\left( \sigma(\mathbf{w}_k \cdot \mathbf{x}^{(i)} + b_k) \right)
\end{align}
$$

where $$\hat{y}_{i,k}$$ is the predicted probability that sample $$i$$ belongs to class $$k$$, and $$y_{i,k}$$ is the true class indicator (1 if sample $$i$$ belongs to class $$k$$, 0 otherwise).

Can you further simplify above equation for the loss function? 
{: .notice--info}


### Python Code Example: Logistic and Softmax Regression

Here's a simple example using `scikit-learn` for both binary logistic regression and multiclass softmax regression:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_multilabel_classification
import matplotlib.pyplot as plt

# Binary classification with logistic regression
X_bin, y_bin = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
model_bin = LogisticRegression()
model_bin.fit(X_bin, y_bin)

# Multiclass classification with softmax regression
X_multi, y_multi = make_multilabel_classification(n_samples=100, n_features=2, n_classes=3, random_state=42)
model_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model_multi.fit(X_multi, y_multi.argmax(axis=1))

# Plotting decision boundaries for binary logistic regression
plt.scatter(X_bin[:, 0], X_bin[:, 1], c=y_bin, cmap='bwr', label='Data points')
x_vals = np.linspace(X_bin[:, 0].min(), X_bin[:, 0].max(), 100)
boundary_bin = lambda x: (-model_bin.intercept_[0] - model_bin.coef_[0][0] * x) / model_bin.coef_[0][1]
plt.plot(x_vals, boundary_bin(x_vals), color='black', label='Decision boundary')
plt.title('Binary Logistic Regression')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Plotting decision boundaries for softmax regression (multiclass)
plt.scatter(X_multi[:, 0], X_multi[:, 1], c=y_multi.argmax(axis=1), cmap='viridis', label='Data points')
plt.title('Softmax Regression (Multiclass)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

Logistic Regression and Softmax Regression extend linear models to classification problems. Logistic regression handles binary classification by estimating probabilities using the logistic function while softmax regression extends this concept to handle multiple classes using the softmax function. Both are integral to understanding and implementing simple yet powerful classification models in machine learning.


