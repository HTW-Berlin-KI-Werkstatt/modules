---
title: "Numpy essentials"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

NumPy is a fundamental package for scientific computing in Python. It provides support for arrays, matrices, and many mathematical functions to operate on these data structures efficiently. 
The following code examples assume that you have python running in an virtual environment with ``numpy``, ``scikit-learn``, and ``matplotlib`` installed.

The following examples are of course just the tiny fraction of the top 0.1% of the tip of an iceberg. You need to learn about further functions yourself on the fly during the lecture.

## Starting with numpy

Let's get started by the following example:
```python
import numpy as np

# Creating a NumPy array
array = np.array([1, 2, 3, 4, 5])
print("Array:", array)

# Performing basic operations
print("Sum:", np.sum(array))
print("Mean:", np.mean(array))
print("Standard Deviation:", np.std(array))
```

The type ``np.array`` is the standard data type of numpy for vectors, matrices, and tensors (more than 2 indices).
The object ``array`` has a certain shape:
```python
array.shape # returns (5,)
```
in our case it has 1 axis (1 index used to identify elements) with 5 dimensions (index ranges from 0 to 4).
All the operations above (sum, mean, std) can be also restricted to certain axis, for example, if you want to compute the mean of each column in one operation.

It takes some time to get used to addressing the right elements in numpy arrays and optimizing the code with *vectorization*, i.e. addressing and
operating with multiple elements at once.

Scikit-learn provides several datasets that are ready to use with NumPy. 
Here, we'll use the Iris dataset as an example to demonstrate how to load and manipulate data with NumPy:
```python
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target

print("Data shape:", data.shape)
print("Target shape:", target.shape)
```

Let's use NumPy to explore the dataset:
```python
# Importing NumPy
import numpy as np

# Calculate the mean of each feature
mean_features = np.mean(data, axis=0)
print("Mean of each feature:", mean_features)

# Calculate the standard deviation of each feature
std_features = np.std(data, axis=0)
print("Standard deviation of each feature:", std_features)

# Calculate the correlation matrix
correlation_matrix = np.corrcoef(data.T)
print("Correlation matrix:\n", correlation_matrix)
```

## Visualizing Data with Matplotlib

For a better understanding of the data, let's plot some features using Matplotlib, a popular plotting library.
Matplotlib is by far not the most modern library and not the fanciest one, but a solid start. 
Let's create a simple scatter plot of the first two features in the iris dataset:

```python
import matplotlib.pyplot as plt

# Scatter plot of the first two features
plt.scatter(data[:, 0], data[:, 1], c=target, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris Dataset: Sepal length vs Sepal width')
plt.colorbar(label='Species')
plt.show()
```

Please note the wonderful application of vectorization when providing the data to the scatter command.

## Explore further

1. Learn how to generate random matrices of arbitrary shape with ``np.random.rand`` and ``np.random.randn``!
2. What are other datasets available in scikit-learn?
3. Learn how to do matrix multiplications with ``np.dot``!
4. Enjoy the multi-index magic of numpy with statements like ``data[target==0, 0]``!

## Conclusions
In this section, we have introduced the basics of NumPy, demonstrated how to use it with the Iris dataset from scikit-learn, and visualized the data using Matplotlib.
There are endless possibilities with numpy, which are best explored on the fly once you need them.