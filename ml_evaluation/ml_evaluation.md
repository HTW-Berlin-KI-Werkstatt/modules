---
title: "Evaluating machine learning models"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Evaluation is a core aspect of machine learning. Therefore, we take the unusual step to talk first about evaluation and then later on we
go into more detail what kind of models are suitable and how to do the learning. 
Proper evaluation ensures that the model performs well not only on the training data but also on unseen data.

## 1. Never Test on Training Data

One of the fundamental principles in evaluating machine learning models is to **never test on the training data**. Testing on training data can lead to overestimating the model's performance because the model has already "seen" this data. Instead, we should evaluate the model on separate data that was not used during training to obtain an unbiased estimate of its true performance.
Remember the previous lessons and the definition of *overfitting*. Testing on training data would never spot overfitting errors, instead we would get a very optimistic
estimate of the error of the model, but this would not reveal any information about generalization.

Testing your knowledge by trying to answer questions you already memorized is also not a good idea and likely gives you a wrong feeling for
your current competencies.

## 2. Importance of Randomness and Repetitions

Another important aspect of evaluation is to design an evaluation that is robust with respect to the selection of training and test data.
Your selection of the training data from the overall dataset might be biased and your estimate of the performance of the model is skewed since you should
by chance selected a non-representative part of the data for training. 

Therefore, two aspects need to be included for evaluation:
- **Randomness:** Shuffling the data before splitting the dataset into training and testing helps to ensure that the subsets are representative of the overall data distribution.
- **Repetitions:** Running multiple rounds of cross-validation or repeating the splitting process with different random seeds can provide a more accurate estimate of model performance. Furthermore, you can estimate the standard deviation of the performance metrics. Getting a 100% accuracy in one trial and a 50% chance-level accuracy in another trial might not be a good sign for predictions on future data.

## 3. Splitting Strategies for the Data

Let's look at some evaluation and especially splitting techniques to deal with the aspects explained in the last paragraphs.

### 3.1 Cross-Validation

Cross-validation is a robust technique for assessing the performance of a machine learning model. The most common form is k-fold cross-validation:
- **k-Fold Cross-Validation:** The dataset is split into $$k$$ equal parts. The model is trained on $$k-1$$ parts and tested on the remaining part. This process repeats $$k$$ times, with each part being used exactly once as the test set. The final performance metric is the average of the results from all `k` iterations.

The following code snipplet which skips some parts such as data loading shows how to do $$k$$-fold cross-validation in python:
```python
from sklearn.model_selection import KFold, cross_val_score
model = SomeModel()
kf = KFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=kf)
print("Cross-validation scores:", scores)
```

In this case the function ``cross_val_score`` takes care of the training and the evaluation.
In the majority of cases, you might want to have more control over this process and loop yourself over the splits provided by ``kf``.

Please also read the excellent tutorial at [https://mlu-explain.github.io/cross-validation/](mlu-explain) on
cross-validation.

### 3.2 Train/Validation/Test Split

Another common strategy is to split the dataset into three distinct sets: 
- **Training Set:** Used to train the model.
- **Validation Set:** Used for tuning hyperparameters and model selection (we will come to this later).
- **Test Set:** Used for the final evaluation of the model.

A typical split might be 60% training, 20% validation, and 20% testing, but this very much depends on the size of the dataset:
```python
from sklearn.model_selection import train_test_split
# split into train and temp first
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
# split temp into val and test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

Please also read the excellent tutorial at [https://mlu-explain.github.io/train-test-validation/](mlu-explain) on
splitting strategies.

## 4. Metrics for Regression

So far we talked about evaluation strategies (just covered the tip of the iceberg here), but how do we actually compare
our prediction $$y_i$$ (for each test example with index $$1 \leq i \leq n$$) with the ground-truth labels $$\hat{y}_i$$?
This depends on many aspects, first of all let's focus on regression tasks where $$y$$ is a continuous value.

### 4.1 Mean Absolute Error (MAE)

A very standard measure is the absolute difference between both values, which we can averaged to obtain
a single value for the whole dataset:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \vert\hat{y}_i - y_i\vert$$.

 
**Pros:**
- Easy to understand and interpret.
- Less sensitive to outliers compared to MSE.

### 4.2 Mean Squared Error (MSE)

Mathematicians since Gauss prefer the squared error:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2,$$

which is especially suitable for certain mathematical derivations.
Related to MSE, you can often find the root mean squared error (RMSE) in the literature and guess what, it is simply the
root of the MSE, but has the same unit as the input value and therefore is easier to interpret.

**Pros:**
- Heavily penalizes larger errors due to squaring, which can be useful in emphasizing significant discrepancies.
- Easy to differentiate (we come to this later)

**Cons:**
- More sensitive to outliers as large errors are squared.

### 4.3 R-squared (R²)

Let $$\bar{y}$$ be the mean of the ground-truth values (across the test set), then we can compute the so-called $$R$$ square metric:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (\hat{y}_i - y_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

The denominator should look familiar: it is the variance of the labels without normalizing it wrt. to the number of test examples.
This measure normalizes the overall MSE error with the variance of the ground-truth labels. In which situations do you think this is reasonable?

**Pros:**
- Provides a measure of how well the model explains the variance in the data.
  
**Cons:**
- Can be misleading for non-linear models; a high $$R^2$$ doesn't always indicate a good model.

These metrics are easy to obtain in python:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)
```

## 5. Metrics for Classification

In contrast to regression, we need other metrics for classification where the output is a discrete value $$y \in \{0, \ldots, K-1\}$$ representing
one of $$K$$ classes.


### Confusion matrix

To understand the next metrics and to get detailed insights into classification results, a key tool is the **confusion matrix**.
When confronted *with only two classes* in the task, a confusion matrix consists of four components:

- **True Positives (TP):** The number of correct positive predictions.
- **True Negatives (TN):** The number of correct negative predictions.
- **False Positives (FP):** The number of incorrect positive predictions, also known as Type I errors.
- **False Negatives (FN):** The number of incorrect negative predictions, also known as Type II errors.

The confusion matrix can be represented in a tabular format:

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| Actual Positive| TP                 | FN                 |
| Actual Negative| FP                 | TN                 |

and allows to derive many different metrics explained in the following.

In the following, we will go through several performance metrics for classification.
There is also a great tutorial at [mlu-explain](https://mlu-explain.github.io/precision-recall/)
worth reading.

#### Accuracy

The most common metric that people use (and fail to use properly) is accuracy:

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} = \frac{1}{n} \sum\limits_{i=1}^n [ \hat{y}_i = y_i ] = \frac{TP + TN}{TP + TN + FP + FN}$$

What is the percentage of test examples with the correctly predicted label? In the formula above $$[ z ]$$ is 1 if $$z$$ is true and 0 otherwise.
This naturally applies to scenarios with $$K>2$$ as well.

**Pros:**
- Simple to understand and compute.
- Works well with balanced datasets.

**Cons:**
- Misleading for imbalanced datasets as it may give high accuracy by simply predicting the majority class: Assume we have a test set that consists of 990 images without a pedestrian and
10 images with a pedestrian. A model simply predicting "no predestrian" would get a 99% accuracy. Time to raise some venture capital 😄.

#### Recall (alternative name: Sensitivity)

![Recall and precision](img/precision_recall.svg)
> Visual explanation of recall and precision (image from [wikipedia](https://commons.wikimedia.org/wiki/File:Precisionrecall.svg))

Another common measure only concentrates on positives examples:
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
This is also known as the accuracy of the positive class only: how many positive examples have been recognized correctly?
Please note that the denominator is simply the number of positive examples in the test set.
This measure ignores that the number of false positives might be really high.

Taking only the negative class into account, we get the performance metric of specificity. This is 
easily confused with sensitivity.

#### Precision

Similar to recall, precision involves the number of true positives in the test set - all positive examples
that have been correctly recognized. However, precision uses the number of all examples that were
predicted as being positive for normalization:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

If the number of false positives is high, precision drops significantly.

* Pro: Useful when the cost of false positives is high.
* Con: Should be considered alongside recall for a comprehensive assessment.


#### Further measures
* Specificity: $$\text{Specificity} = \frac{TN}{TN + FP}$$
* F1 measure: $$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  * Pro: Balances precision and recall and useful for imbalanced datasets
  * Con: Can be less interpretable compared to precision and recall individually.


### Example Code to Compute Confusion Matrix

Here's an example using Python's Scikit-learn library to compute and visualize the confusion matrix:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Assume y_true and y_pred are your true labels and predictions respectively
y_true = [0, 1, 0, 1, 0, 1, 1]
y_pred = [0, 0, 0, 1, 1, 1, 0]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
```

### ⭐Receiver Operator Characterics

The **Receiver Operating Characteristic (ROC) Curve** is a graphical plot used to evaluate the performance of a binary classification model that not only provides discrete decisions but also a continuous value (score) related to the likelihood of an example being positive. 
To get to a discrete decision, the score of such a model is thresholded, e.g. if the score is above $$T=0.5$$, the example is considered to be predicted as positive. 
However, we could of course choose another threshold resulting in different decisions. A higher threshold $$T$$ obviously results
in a equal or smaller number of false positives. In some cases, we rather make careful decisions and only consider the example to be positive, if the score
is really high. True positive rate and false positive rate therefore depend on $$T$$, and a ROC curve simply plots $$(\text{FPR}(T), \text{TPR}(T))$$.

![ROC curve](img/roc.webp)
> Infographic by [İlyurek Kılıç](https://medium.com/@ilyurek/roc-curve-and-auc-evaluating-model-performance-c2178008b02)

In essence, we get a visualization of all possible trade-offs by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
It simply depends on the application, whether you want to have a model being careful about predicting something to be positive or not.

A great animated explanation for ROC curves can be also found at [https://mlu-explain.github.io/roc-auc/](mlu-explain).

### How to Interpret the ROC Curve

- **Starting and end point**: A ROC curve always starts at $$(0,0)$$ (all decisions are negative) and ends with $$(1,1)$$ (all decisions are positive).
- **Diagonal Line:** A ROC curve that lies on the diagonal line from the bottom-left to the top-right represents a random guess, indicating no discrimination capacity by the model.
- **Above Diagonal Line:** The closer the ROC curve to the top-left corner, the better the model is at distinguishing between the two classes.
- **Area Under the Curve (AUC):** The area under the ROC curve (AUC) quantifies the overall ability of the model to distinguish between positive and negative classes. An AUC value of 1 represents a perfect classifier, whereas an AUC value of 0.5 represents a worthless classifier.

### Example Code to Plot ROC Curve

Here's an example using Python's Scikit-learn library to compute and visualize the ROC curve:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Assume y_true are the true labels and y_scores are the predicted probabilities
y_true = [0, 1, 0, 1, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Compute AUC
roc_auc = roc_auc_score(y_true, y_scores)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

In binary classification scenarios, you should always use ROC curves or related techniques instead of single performance measures.
{: .notice--warning} 