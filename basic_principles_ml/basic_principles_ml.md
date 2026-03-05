---
title: "Basic machine learning principles"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

As already covered in the introductory lecture, unlike traditional programming, where specific instructions are coded, 
machine learning models learn patterns from data and make predictions on new data. We will mainly cover supervised learning in the lecture, since
this is the main branch of methods in machine learning that has been successfully applied to a wide range of applications.

## Supervised Learning

In supervised learning, the model is trained on a labeled dataset, meaning that each training example is paired with an output label. These output labels
of the training set are most of the time provided by human annotators. 

![](img/mnist.png)

The image shows a part of the famous [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset for letter recognition. Each of the images displayed is labeled
according to the number shown in the image. Labeling, often also referred to as annotation, can be an expensive and hideous task. 

With these datasets, the goal is to learn a mapping, often referred to as a model, from inputs $$\mathbf{x}$$ to outputs $$y$$ that can be used to make predictions on new, unseen data. In our MNIST example case, an input 
$$\mathbf{x}$$ is an image, and the output $$y$$ would be a number from 0 to 9.

### Important terms in supervised learning

1. **Model:** A model is a mathematical representation of a function or algorithm. 
2. **Training Set:** The training set is a subset of the dataset used to train the model. It consists of input-output pairs where the output (label) is known.
3. **Test Set:** The test set is a separate subset of the dataset used to evaluate the model's performance. The model makes predictions on the test set, and these predictions are compared to the actual (human-provided) labels to assess accuracy.
4. **Features:** Features are individual measurable properties or characteristics of the data. In a dataset, features are the inputs to the model. We will come to this later.
5. **Labels:** Labels are the outputs or target values in a supervised learning problem. 

## Typical Tasks in Supervised Learning

There are two main tasks in supervised learning: classification and regression. Let's look into them with some examples and applications:

### Classification

In classification, the goal is to **predict a categorical label** $$y \in \{0, \ldots, K-1 \}$$. The model assigns inputs to one of several predefined categories. Examples of classification tasks include:

- **Spam Detection:** Classifying emails as spam or not spam ($$K=2$$).
- **Image Recognition:** Identifying objects in images, such as distinguishing between different animals ($$K$$ is the number of species you want to differentiate).
- **Sentiment Analysis:** Determining the sentiment of a piece of text as positive, negative, or neutral ($$K=3$$).

What are other applications for classification you can think of?
{: .notice--info} 


Let's do some first serious coding and get a classification model running:
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create ...
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# 3. ...  and train the classifier
classifier.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = classifier.predict(X_test)

# 5. Compare model predictions with ground truth
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

This example shows a classical machine learning pipeline:
1. Load and preprocess the data (there was no preprocessing involved in the code above).
2. Split the given data into a training and a test set - can you figure out how the splitting was done? Why is there a random seed value fixed?
3. Train (fit) the model - there is a random forest classifier used here, but let's not get into detail already.
4. Get all predictions for the test set (predicted classes) - why do we need to split the data anyway?
5. Compare the predictions of the model with the so-called ground truth (the human labels).

The above model operates on the iris dataset, which is a tabular and not an image dataset.

### Regression

In regression, the goal is to **predict a continuous value** $$y \in \mathbb{R}$$. The model learns the relationship between input features and a numerical output. Examples of regression tasks include:

- **House Price Prediction:** Predicting the price of a house based on its features, such as size, location, and number of bedrooms.
- **Weather Forecasting:** Predicting temperature, humidity, or other weather parameters.
- **Stock Price Prediction:** Estimating the future price of a stock based on historical data.

Again, let's do some serious coding:
```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Boston housing dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the regressor
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

The code looks familiar, doesn't it? The main structure of the code is the same. 

Can you spot the differences between the code used for classification and the one used for regression?
{: .notice--info} 

