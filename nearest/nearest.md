---
title: "Nearest neighbour classifiers"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

If two inputs have similar feature vectors, they probably have the same label. Nearest neighbour methods take this idea literally:
"training" only means storing the training set, and for a new input we look up the most similar training examples and copy their labels.
All we need is a [feature vector](/modules/featurevectors/featurevectors.md) for each example and a distance between vectors, as introduced in the [math essentials](/modules/math_essentials/math_essentials.md#linear-algebra).

## Nearest Neighbor Classifier

The Nearest Neighbor classifier is a simple, intuitive approach to classification that assigns a data point $$\mathbf{x}$$ the label of its closest training example.

Let us be given a set of training examples $$\{(\mathbf{x}^{(i)}, y_i)\}_{i=1}^N$$, where $$\mathbf{x}^{(i)} \in \mathbb{R}^D$$ are the inputs and $$y_i \in \{1, 2, \ldots, C\}$$ are the class labels.
For a new input $$\mathbf{x}$$, we find the nearest neighbor and its label by:

$$ \hat{y} = y_j \quad \text{where} \quad j = \arg\min_i \lVert \mathbf{x} - \mathbf{x}^{(i)} \rVert^2 $$

The notation $$\lVert \cdot \rVert$$ is used here for the norm of the vector. The last part of the equation is therefore simply
the squared Euclidean distance of the test example $$\mathbf{x}$$ and the training example $$\mathbf{x}^{(i)}$$.

Every training example "owns" the region of all inputs that are closer to it than to any other training example (its [Voronoi cell](https://en.wikipedia.org/wiki/Voronoi_diagram)).
The decision boundary of the classifier therefore consists of pieces of the borders between cells of examples with different labels.

**Try it yourself:** the [kNN playground](https://ml.kanaries.net/playground/knn) runs in your browser. Add training points, drag the query point around, and watch how the prediction and the neighbours change.
{: .notice--success}

## k-Nearest Neighbors

A single neighbour can be misleading, for example if its label is wrong or it is an outlier inside the region of another class.
The $$k$$-nearest neighbour classifier (k-NN) therefore looks at the $$k$$ nearest training examples $$\mathcal{N}_k(\mathbf{x})$$ and predicts by majority vote:

$$ \hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(\mathbf{x})} [y_i = c] $$

The bracket $$[y_i = c]$$ is 1 if the statement inside is true and 0 otherwise, so the sum simply counts the neighbours of class $$c$$.
For $$k=1$$, we get the nearest neighbour classifier from above.
Ties can happen if $$k$$ is even; for two classes, an odd $$k$$ avoids them, otherwise one can fall back to the label of the single nearest neighbour.
The fraction of neighbours with class $$c$$ can also be used as a (rough) estimate of the probability $$p(y=c \mid \mathbf{x})$$.

### Choosing k

The number of neighbours controls how smooth the decision boundary is:

![k-NN decision regions for k=1 and k=15 on the same noisy two-class dataset](/modules/nearest/images/knn_k_comparison.svg)

With $$k=1$$, every training example is classified correctly (each one is its own nearest neighbour), but the boundary follows every noisy example and builds small islands.
With $$k=15$$, the boundary is much smoother and ignores single outliers, at the cost of some training errors.
If we go to the extreme $$k=N$$, all training examples vote and the classifier always predicts the most frequent class.

The training accuracy is therefore useless for choosing $$k$$. Instead, $$k$$ is a hyperparameter that we select on a validation set or with cross-validation, as discussed in the [evaluation module](/modules/ml_evaluation/ml_evaluation.md).

Can you think of other strategies for making the prediction based on the $$k$$ nearest neighbors?
{: .notice--warning}

### Things to watch out for

**Scaling of features.** The distance treats all dimensions equally. If one feature is measured in euros (values up to 100,000) and another one in years (values up to 100), the first feature dominates the distance and the second one is practically ignored.
It is therefore common to rescale every feature first, for example to zero mean and unit variance.
We will meet this problem again in [lecture 05](/modules/linear_model_weights/linear_model_weights.md), where standardization and normalization of features are discussed in more detail.

**High dimensions.** For uniformly distributed random points, the distances between all pairs of points become more and more similar as the dimension grows, so "nearest" loses its meaning.
This is one aspect of the *curse of dimensionality*, which you can explore in this [interactive guide](https://www.gm.th-koeln.de/~bartz/bart21i/en_curse_interactive.html).
Real data, such as images or texts turned into embeddings, does not fill the space uniformly, which is why nearest neighbours still work well with good feature vectors.

**Prediction time.** For every prediction, we need to compute the distance to all $$N$$ training examples. This brings us to the next section.

## Fast nearest neighbours

Computing all $$N$$ distances for each query is fine for a few thousand examples, but too slow for millions of images or text embeddings.
Fast methods avoid most of these distance computations, often by accepting that they return only an *approximately* nearest neighbour.

**kd-trees.** A kd-tree splits the data recursively along one coordinate at a time. A query first descends to its leaf and then only checks those other cells of the tree that could still contain a closer point.
This gives large speed-ups in low dimensions, but in high dimensions almost all cells have to be checked again.
Interactive demo: [KD-tree demo (Cornell CS4780)](https://www.cs.cornell.edu/courses/cs4780/2022sp/notes/kdtree.html) – hover over the plot to see which distance computations are saved.
More: [animations of kd-tree searches by Andrew Moore](https://www.cs.cmu.edu/~awm/animations/kdtree), [lecture notes on kd-trees and ball trees](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html), and the [algorithms used in scikit-learn](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbor-algorithms).

**Binary codes.** Each vector is mapped to a short bit string such that similar vectors get similar codes, for example by checking on which side of several random hyperplanes it lies ([locality-sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)) or with learned hash functions.
Comparing two codes (Hamming distance: count the differing bits) takes only a few processor instructions, and the codes can also be used directly as addresses in hash tables.
More: [Torralba, Fergus, Weiss: Small codes and large image databases for recognition (CVPR 2008)](https://people.csail.mit.edu/torralba/publications/cvpr2008.pdf), which searches millions of images with codes of a few hundred bits, and [Weiss, Torralba, Fergus: Spectral hashing (NIPS 2008)](https://papers.nips.cc/paper/3383-spectral-hashing).

In practice, libraries such as [FAISS](https://github.com/facebookresearch/faiss) ([Johnson et al.](https://arxiv.org/abs/1702.08734)) are used to search millions to billions of embeddings, and [ANN-Benchmarks](https://ann-benchmarks.com/) compares the speed and accuracy of many such libraries.

## Example

Whereas this is a classifier that can be easily written from scratch in 2 minutes, we can also make use again of `scikit-learn`:

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Sample data
X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[0.5, 1]])

# Create and fit K-Nearest Neighbor model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict class for new data point
y_pred = knn.predict(X_test)
print(f"Predicted class for {X_test}: {y_pred}")
```

## Test yourself

**Exercise 1 (distances):** Why is the squared Euclidean distance used instead of the non-squared version? Can you think of other distance measures that might be suitable?
{: .notice--warning}

<details class="notice--info" markdown="1">
<summary><strong>Solution</strong></summary>

The square root is monotonically increasing, so the example with the smallest squared distance also has the smallest distance: the result is the same and we save computing a square root for every training example.

Other options are, for example, the cosine distance from the [math essentials](/modules/math_essentials/math_essentials.md#linear-algebra) (common for embeddings, where the direction matters more than the length), the Manhattan distance $$\sum_d \lvert x_d - x^{(i)}_d \rvert$$, or the Hamming distance (number of differing entries) for binary or categorical features.
</details>

**Exercise 2 (choice of k):** We are given five training examples in 2D:
$$\mathbf{x}^{(1)} = (1,2)$$ with class 1, $$\mathbf{x}^{(2)} = (2,2)$$ with class 2, $$\mathbf{x}^{(3)} = (0,0)$$ with class 2, $$\mathbf{x}^{(4)} = (3,3)$$ with class 1, and $$\mathbf{x}^{(5)} = (4,1)$$ with class 1.
Predict the class of $$\mathbf{x} = (1,1)$$ with $$k=1$$, $$k=3$$ and $$k=5$$.
{: .notice--warning}

<details class="notice--info" markdown="1">
<summary><strong>Solution</strong></summary>

The squared distances to $$\mathbf{x}$$ are 1, 2, 2, 8 and 9.

* $$k=1$$: the nearest neighbour is $$\mathbf{x}^{(1)}$$, so the prediction is class 1.
* $$k=3$$: the neighbours are $$\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \mathbf{x}^{(3)}$$ with classes 1, 2, 2, so the prediction is class 2.
* $$k=5$$: all examples vote (three times class 1, twice class 2), so the prediction is class 1. With $$k=N$$, the prediction is always the most frequent class, no matter where $$\mathbf{x}$$ is.
</details>

## Further reading

* [CS231n notes on image classification with nearest neighbours](https://cs231n.github.io/classification/)
* [Cornell CS4780 lecture notes on k-NN](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html)
* [scikit-learn user guide: nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
