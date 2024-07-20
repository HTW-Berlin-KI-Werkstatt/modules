---
title: "Nearest neighbour classifiers"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

In the following, we look into the simplest ML model that exists - nearest neighbour classifiers.

## Nearest Neighbor Classifier

The Nearest Neighbor classifier is a simple, intuitive approach to classification that assigns a data point $$\mathbf{x}$$ the label of its closest training example(s).

Let us be given a set of training examples $$\{(\mathbf{x}^{(i)}, y_i)\}$$, where $$\mathbf{x}^{(i)} \in \mathbb{R}^D$$ are the inputs and $$y_i \in \{1, 2, \ldots, C\}$$ are the class labels.
For a new input $$\mathbf{x}$$, we find the nearest neighbor and its label by:

$$ \hat{y} = y_j \quad \text{where} \quad j = \arg\min_i \vert \mathbf{x} - \mathbf{x}^{(i)} \vert^2 $$

The notation $$\vert \cdot \vert$$ is used here for the norm of the vector. The last part of the equation is therefore simply
the quadratic distance of the test example $$\mathbf{x}$$ and the training example $$\mathbf{x}^{(i)}$$

Why is the squared Euclidean distance used instead of the non-squared version? Can you think of other distance measures
that might be suitable?
{: .notice--info}

A straightforward extension of the Nearest Neighbor classifier is the $k$-Nearest Classifier: 
we find the $k$ nearest neighbors and then predict according to a majority vote of them.

Can you think of other strategies for making the prediction based on the $k$ nearest neighbors?
{: .notice--info}

## Example

Whereas this is a classifier that can be easily written from scratch in 2 minutes, we can also make use again of `scikit-learn`:

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Sample data
X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[1, 2]])

# Create and fit K-Nearest Neighbor model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict class for new data point
y_pred = knn.predict(X_test)
print(f"Predicted class for {X_test}: {y_pred}")
```

