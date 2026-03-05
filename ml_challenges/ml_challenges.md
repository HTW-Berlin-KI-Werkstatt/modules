---
title: "Challenges in Machine Learning"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

In this module, we will discuss some of the key challenges when dealing with machine learning, including overfitting, data bias, distribution shift, and label noise.

## Data requirements for machine learning

Machine learning requires data, not just some arbitrary samples, but data of a certain quality.
We already learned that supervised machine learning requires not only given input data (e.g. an image) but also labels attached to these inputs for learning.
Furthermore, to train robust and accurate machine learning models, it is essential that the input data adheres to several key requirements. 

Firstly, the data must be **representative** of the real-world scenarios where the model will be applied. This means it should cover a diverse range of conditions and variations inherent in the task at hand, ensuring that the model can generalize well to unseen instances. We therefore require the data to be accurately and consistently collected to avoid biases and inconsistencies that could skew the model's predictions. For example, if you want to detect pedestrians for autonomous driving, you should not only provide daylight images to the model for learning; otherwise, the model will not generalize well during nighttime. 

But wait, we just learned another term: **generalization**. In machine learning, everything is about generalization. We do not want to have stupid models that
just memorize training data, we want the model to generalize, i.e., to deal with new examples that are from the same distribution as the training data but not exactly identical. 

Secondly, the dataset must be **comprehensive enough to capture all relevant information** related to the specific task, including any nuances and subtleties that could impact the outcome.
If there is no information in input $$\mathbf{x}$$, there is no information in input $$\mathbf{x}$$. A machine learning model will not be able to magically squeeze out information
from the examples that are not present (under the assumption that the model does not have access to other information sources or assumptions).

## Overfitting

These thoughts directly bring us to the concept of **overfitting**. It occurs when a machine learning model learns not only the underlying pattern in the training data but also the noise and outliers, e.g. by simply memorizing everything without generalizing. This leads to inferior performance on unseen data, although the model completely fits on the training data sets.

![overfitting model](img/overfitting.png)
> Infographic by [Jen Looper](https://twitter.com/jenlooper)

The image illustrates the phenomenon. Overfitting can also be observed in everyday life. For example, think about your programming lectures. If you just learned from 
the given slides and memorized only code snippets from the lecture, you would not be able to succeed in the exam when being confronted with new tasks. Never overfit your brain, learn by finding patterns instead of memorizing 😄.

- **Symptoms**: High accuracy on training data but low accuracy on validation/testing data.
- **Causes**: 
  - Model complexity (too many parameters).
  - Insufficient training data.
- **Solutions** (we will come to this later):
  - Use simpler models or fine-tuning.
  - Apply regularization techniques.
  - Gather more diverse training data.


## Data Bias

**Data bias** arises when the training dataset does not accurately represent the real-world scenario it is meant to model. 
This can lead to models that are systematically wrong in certain ways. Data bias is also the main aspect working against AI fairness. 
For example, ChatGPT is massively biased, since the dataset contains all the nasty biases of our current reality: racial bias, gender bias, bias with respect
to Western cultures, etc. (see the long list of publications about the topic [here](https://scholar.google.com/scholar?hl=de&as_sdt=0%2C5&as_vis=1&q=Data+Bias+GPT&btnG=)).

- **Types of Bias**:
  - **Selection Bias**: Non-random sample of data, e.g. just using the first 100 items of the dataset for training and then realizing that they are ordered by time and therefore only cover a small part of the day.
  - **Measurement Bias**: Systematic error in data collection.
- **Consequences**: Models with biased predictions harming fairness and equity.
- **Mitigation Strategies**:
  - Ensure diverse and representative data collecting practices.
  - Use debiasing algorithms and techniques.
  - Perform thorough exploratory data analysis (EDA).

### Data bias explained in terms of probabilities

Consider a binary classification problem where $$y$$ is the label, which can be either 0 or 1, and $$\mathbf{x}$$ is the input. 
Let's assume we are building a model to classify whether an email is spam ($$y = 1$$) or not spam ($$y = 0$$).

Suppose the prior distribution of $$p(y)$$ in our training data is heavily skewed: 90% of the emails are labeled as not spam ($$y = 0$$), and only 10% are labeled as spam ($$y = 1$$). This imbalance means that the model will see far more examples of non-spam emails than spam emails during training.

This bias in the prior distribution can lead the model to become overly confident in predicting that an email is not spam, simply because it encounters this label far more frequently. As a result, the model may underperform on spam detection, missing many actual spam emails.

For instance, if our dataset comprises 9,000 non-spam emails and 1,000 spam emails, the prior probabilities are $$p(y = 0) = 0.9$$ and $$p(y = 1) = 0.1$$. A naive classifier that always predicts the majority class (non-spam) would achieve a 90% accuracy, despite failing to identify any spam emails, demonstrating the superficial success of such a biased approach.

To address this issue, techniques such as resampling the dataset (either by undersampling the majority class or oversampling the minority class) or utilizing methods like cost-sensitive learning can be applied. These strategies help ensure that the model is trained on a more balanced representation of the labels, thus mitigating the effects of bias in the prior distribution and improving overall performance on both classes.

## Distribution Shift

**Distribution shift** refers to the change in data distribution between the training and deployment phases. Models trained on one distribution often perform poorly when the input data distribution shifts and maybe also continues to shift over time. Just think about training a model for customer preference, i.e., for a certain product $$\mathbf{x}$$ we are trying to predict whether the customer
is likely to buy the product. By the way, this is all about estimating $$p(y \vert \mathbf{x})$$ if you like to think in stochastic terms. 
However, the likelihood of buying a certain product naturally changes over time depending on what I bought already and the season, for example. 
Just training one model and making predictions with it, therefore, does not work.

![overfitting model](img/topex-visda-examples.png)
> Two examples for a shift of the data distribution [(Ritter et al. 2023)](https://arxiv.org/abs/2310.04757)

- **Types of Distribution Shift**:
  - **Covariate Shift**: Change in the input features' distribution $$p(\mathbf{x})$$ (change of the camera image over time and season for object detection). An example of a covariate shift is given in the image above for two different application scenarios.
  - **Label Shift**: Change in the output label distribution $$p(y \vert \mathbf{x})$$ changes. 
- **Detection**: Statistical tests that compare the distributions of the data over time.
- **Handling Techniques**:
  - Re-train models with new data.
  - Adapt models using continuous learning.
  - Monitor live performance and update periodically.

## Label Noise

**Label noise** is the presence of incorrect or inconsistent labels in the training dataset. This can degrade the model's performance by providing inaccurate signals during training.
Some tasks are even so hard that label noise is a natural thing to deal with and cannot be avoided. This is, for example, the case for medical applications - ask $$N$$ doctors to annotate the data, and you get $$N$$ different answers.

- **Sources**:
  - Human annotation errors.
  - Ambiguities in data labeling, i.e., it's simply hard.
- **Effects**: Reduced accuracy, unreliable predictions.
- **Approaches to Address**:
  - Try to identify and correct noisy labels (what are the examples people heavily disagree on?).
  - Use machine learning models (and loss functions) that can deal with it.
  - Review your annotation guidelines and process.
