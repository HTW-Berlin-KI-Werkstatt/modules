---
title: "Automated Machine Learning (AutoML)"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

We learned already quite a lot about different models, their hyperparameters, their variants and preprocessing involved. But what kind of model should I use? AutoML comes to the rescue here.
Automated Machine Learning (AutoML) is a rapidly growing field aiming to automate and especially optimize the process of applying machine learning to real-world problems. AutoML encompasses the entire pipeline from data preprocessing, feature engineering, model selection, hyperparameter tuning, and model evaluation.

Let's review what we can and should optimize in a machine learning pipeline:
1. **Data Preprocessing**: Automatically handling missing values, encoding categorical variables, scaling features.
2. **Feature Engineering**: Creating new features (using values of existing ones) or selecting the most important ones.
3. **Model Selection**: Choosing the best model architecture for the given task (relevant for neural networks later).
4. **Hyperparameter Tuning**: Optimizing the hyperparameters of selected models.
5. **Model Ensembling**: Combining multiple models to improve performance, referencing our previous lecture on **Ensembles** where techniques like bagging and boosting were discussed.

The benefits of AutoML are pretty obvious:
- **Time-Efficiency**: Reduces the time required to manually tune and select models.
- **Performance**: Often yields high-performing models by exploring a broader range of possibilities than a human might.
- **Accessibility**: Makes machine learning accessible to non-experts.

### AutoML with Python: Using `AutoGluon`

`AutoGluon` is a quite powerful library developed by Amazon that simplifies and accelerates machine learning tasks. It supports various ML tasks, including tabular prediction, text understanding, and image classification.

- **Automatic Feature Engineering**: AutoGluon transforms raw tabular data into features suitable for machine learning models.
- **Model Ensembling**: It automatically trains multiple types of models and ensembles them to boost performance.

Here's an example of using `AutoGluon` for a regression task:

```python
from autogluon.tabular import TabularPredictor

# Load dataset
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

# Specify predictor
predictor = TabularPredictor(label='class').fit(train_data)

# Make predictions
predictions = predictor.predict(test_data)

# Evaluate predictions (assuming ground truth is available)
results = predictor.evaluate_predictions(y_true=test_data['class'], y_pred=predictions)
print(results)
```

You can also specify more advanced settings for model optimization:

```python
predictor = TabularPredictor(
    label='class',
    eval_metric='accuracy',
    presets='best_quality'
).fit(
    train_data,
    time_limit=3600,  # Limit the time for training
    hyperparameters={
        'NN': {},  # Configure neural networks
        'GBM': {'num_boost_round': 100},  # Configure gradient boosting machines
        'CAT': {},  # CatBoost
        'XT': {},  # ExtraTrees
        'KNN': {},  # k-Nearest Neighbors
    }
)
```

As discussed in our previous lecture on ensembles, combining multiple models can significantly improve predictive performance. AutoGluon leverages ensemble methods behind the scenes:
- **Bagging**: Combines predictions from multiple models trained on different subsets of the data.
- **Boosting**: Sequentially trains models, giving more weight to instances that were previously mispredicted.
- **Stacking**: Uses a meta-model to learn how to best combine the predictions of several base models.

### Further Reading and Resources

1. [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
2. [AutoGluon GitHub Repository](https://github.com/awslabs/autogluon)
