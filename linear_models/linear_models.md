---
title: "Linear Models"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

### Linear Models for Machine Learning

Linear models are foundational techniques in machine learning, widely used for both regression and classification tasks. They are essential due to their simplicity, (pseudo-)interpretability, and efficiency.

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

#### Why Linear Models?

Linear models make sense for regression because they assume a straight-line relationship between the dependent and independent variables. This assumption simplifies the computation and makes the model easy to interpret. They work well when:

- The relationship between the features and the target is approximately linear.
- You need an easily interpretable model.
- Speed and performance are critical, and complex models aren't necessary.

Using a linear combination of features to predict the target variable ensures that the model captures the additive effects of each feature. Even if the data is not perfectly linear, linear models provide a good baseline and often serve as a benchmark for more complex methods.

### Application to Classification: Logistic Regression

For classification tasks, linear models can also be very effective. A common approach is **Logistic Regression** for binary classification.


### Python Code Example

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

Solving classification tasks with logistic regression is equally straightforward:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# Initialize and fit the model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities
probabilities = model.predict_proba(X)[:, 1]

# Define decision boundary
boundary = lambda x: (-model.intercept_[0] - model.coef_[0][0] * x) / model.coef_[0][1]

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', label='Data points')
x_vals = np.linspace(-3, 3, 100)
plt.plot(x_vals, boundary(x_vals), color='black', label='Decision boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

### Conclusion

Linear models are fundamental in machine learning due to their simplicity and effectiveness. For regression tasks, they assume a linear relationship between inputs and outputs, making them easy to understand and implement. For classification tasks, logistic regression provides a powerful method to predict class probabilities and define decision boundaries.

By understanding and leveraging linear models, you gain a solid foundation that is essential when exploring more complex machine learning algorithms.