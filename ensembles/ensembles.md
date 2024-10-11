---
title: "Ensembles"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Ensemble methods are techniques that create multiple models and then combine them to produce improved results. The main idea is that a group of weak learners can come together to form a strong learner. 

## Benefits of Heterogeneous Ensembles

The motivation behind using heterogeneous ensembles is that different models may capture different patterns in the data, and combining them can lead to a better overall performance. In the following, we will mathematically justify this intuition.

Let $$\mathcal{X}$$ be a feature space and $$\mathcal{Y}$$ be a label space, where the true relationship between features and labels is described by a function $$g: \mathcal{X} \rightarrow \mathcal{Y}$$. Suppose we have two models $$f_1$$ and $$f_2$$.

Assume that both models have differences to the ground-truth that are normally distributed with mean zero and variance $$\sigma^2$$, 

$$
\begin{align*}
f_1 - g &= \epsilon_1 \sim \mathcal{N}(0, \sigma^2),\\
f_2 - g &= \epsilon_2 \sim \mathcal{N}(0, \sigma^2)
\end{align*}
$$

The squared error of the models is therefore characterized by the variance of $$\epsilon_1$$ and $$\epsilon_2$$
respectively (please remember the definition of the variance here).

Very importantly, we are not assuming $$f_1$$ and $$f_2$$ to be independent here!
In contrast, their correlation is $$\rho$$. If you are asking yourself, what correlation really is, please go back and review your statistics basic! If $$\rho$$ is 0, both models are statistically independent. If $$\rho$$ is close to 1, both
models are highly correlated, i.e. they agree with each other quite often.

Let's get back to ensembles. An ensemble now combines predictions from multiple models.
The simplest combination is by averaging the predictions:
$$f_{\text{ensemble}} = \frac{f_1 + f_2}{2}$$.

Let's now look at the difference to the ground-truth of this ensemble:

$$
\begin{align*}
\epsilon_{\text{ensemble}} &= f_{\text{ensemble}} - g,\\
                           &= \frac{f_1 + f_2}{2} - g,\\
                           &= \frac{\epsilon_1 + \epsilon_2}{2},\\
                           &\sim \mathcal{N}(0, \frac{\sigma^2}{2} \cdot (1 + \rho))
\end{align*}
$$

The last statement might fall from heaven, but it is actually a classical result for the sum of two normally distributed variables (see ["Sum of normally distributed random variables"](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables) and ["Normal distribution"](https://en.wikipedia.org/wiki/Normal_distribution)) for the case on correlated random variables.

We can learn quite a lot from the last statement:
1. The largest error of the ensemble is reached for $$\rho=1$$, i.e. both models outputs agree with each other (positive linear dependance). In this case, we do not gain anything with our ensemble. It's like having a committee of people always agreeing.
2. The smallest error can be achieved with a $$\rho=-1$$ (anti-correlation), i.e. both models do not agree.
3. Independent models in the ensemble lead to a reduction of the error by 50%.

Of course, in reality nothing is perfectly normally distributed and our ensembles involve more than two models. But
similar derivations can be done in way more general settings. In summary, independent or even disagreeing models lead to a strong ensemble.

## Ensemble Methods

Let's look at multiple ways to build an ensemble.

### Bagging
Bagging (Bootstrap Aggregating) is an ensemble method that trains multiple instances of the same model type on different subsets of training data selected via bootstrapping (random selection of examples). This is a super effective and very easy strategy in machine learning.

#### Steps for Bagging:
1. Generate several subsets from the original datasets.
2. Train each subset with a base estimator.
3. Aggregate the predictions from all the base estimators (e.g., through averaging for regression or voting for classification).

In the following example, you can see a classical example of bagging with a decision tree as a base estimator.
This is sometimes referred to as random forest and randomization can be also part of the decision tree training itself:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize base estimator
base_estimator = DecisionTreeClassifier()

# Bagging
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging Test Accuracy: {accuracy:.4f}")
```

Bagging for decision trees is often termed random forest [(Breiman 2001)](https://link.springer.com/article/10.1023/A:1010933404324). There is a great tutorial on random forests at [mlu-explain](https://mlu-explain.github.io/random-forest/).

### Boosting
Boosting is an ensemble method that sequentially trains multiple models, each trying to correct the mistakes of the previous one. In contrast to bagging, models are therefore not trained independently and their decision is also not simply averaged but weighted afterwards.

The main algorithm works as follows:
1. **Initialize model**: Start with a simple model (often referred to as weak learner) trained on the dataset.
2. **Weight adjustment**: After the first model is trained, **adjust the weights of each training instance**. Instances that were misclassified by the model are given higher weights so that subsequent models focus more on these difficult cases.
3. **Iterative training**: **Train** the next weak model **on the weighted data**. This process continues for a set number of iterations or until the model achieves a certain level of performance. Importantly, a model type needs to be used that allows for using instance weights. This can be done for example for decision trees or stumps (single thresholding of single features).
4. **Model combination**: The final prediction is made by combining the predictions from all the models. Typically, these predictions are weighted based on the accuracy of the corresponding models. If a single model has a high accuracy on the training data it should given a high weight in the final decision.



