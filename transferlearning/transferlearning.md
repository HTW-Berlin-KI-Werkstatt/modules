---
title: "Fine-tuning and foundation models"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Understanding fine-tuning is crucial for efficiently building machine learning models, especially when data resources are limited. This lecture will cover how these techniques leverage pre-trained models to enhance performance on new tasks with reduced training time and computational cost. Fine-tuning is a special case of transfer learning, which refers to the use of other but related data to solve a task.

## The Concept of Fine-Tuning

Fine-tuning involves taking a pre-trained model — an architecture trained on one problem — and leveraging it to solve another similar problem. This strategy capitalizes on the knowledge captured in the pre-trained model to improve the learning process of a target task. Why should you use fine-tuning?

1. **Limited Data**: In many scenarios, gathering a large dataset is challenging. Fine-tuning allows the use of smaller datasets by starting from a model pre-trained on extensive datasets.
2. **Reduced Training Time**: Since much of the feature extraction occurs during the original training, adaptation to a new task requires less computational effort.
3. **Performance Enhancement**: Pre-trained models often yield better generalization since they have already learned robust representations from large-scale data.

Fine-tuning is a specific approach within transfer learning where we continue training the pre-trained model on new data. It requires adjusting certain layers while potentially keeping others fixed, balancing between retaining learned features and adapting to new data specifics. But how does fine-tuning work?

1. **Base Model Selection**: Choose a pre-trained model appropriate for your data domain (e.g., ResNet, VGG, Inception).
   
2. **Output Layer Modification**: Replace or modify the output layer to suit the new task's class count.
   
3. **Freeze Initial Layers** (optional): Freeze earlier layers to retain generic features. These layers capture fundamental patterns and typically do not require significant adjustments. This might hold for the first convolutional layers of a CNN.

4. **Train Remaining Layers**: Train only the unfrozen layers with a small learning rate. 
   

## Leveraging a Good Initial Start

Fine-tuning benefits from having a well-placed initial starting point for gradient descent. **The pre-trained model serves as a 'good initial solution'**, helping the optimization process converge more efficiently and avoiding poor local minima. This aspect significantly accelerates training by refining an already effective solution, instead of starting from scratch.

Imagine using VGGNet, pre-trained on ImageNet (a dataset with 1,000 classes), to classify medical images into 10 categories. You would:
- **Replace the final dense layer to output 10 classes**.
- **Freeze early convolutional blocks to preserve basic feature detection**.
- **Unfreeze later layers**, allowing them to learn the nuances of medical imaging.

## Linear probing as a special case

Another strategy to use a pre-trained model is to use it as a feature extractor. We can simply cut the model
at a certain layer $$L$$ and use the output of the model at this layer as a feature vector (often called *learned embedding*).
For our task, we can then simply utilize whatever machine learning model we like to train it on our data.
We have seen this strategy in [our lecture on feature vectos](/modules/featurevectors/featurevectors.md).

If a linear model is applied on the learned embeddings, this approach is sometimes refered to as *linear probing*. In this case, it is a special form of fine-tuning. Where all layers up to $$L$$ are hold fixed and all other layers are replaced with a linear model as the only unfrozen layer to be used.

## Foundation Models

Foundation models represent a paradigm shift in AI, capable of generalizing across diverse tasks. These models serve as a comprehensive starting point for solving various downstream tasks through minimal fine-tuning or prompting. The later
technique is something than we will learn in upcoming lectures.

Foundation models have the following characteristics:

1. **Pre-training on Broad Data**: Foundation models are trained on vast datasets encompassing various domains, enhancing their generalization capabilities - for example a CNN trained on the ImageNet dataset.
   
2. **Scalable Architecture**: These models are designed to extend easily across multiple tasks.
   
3. **Utility Across Domains**: By learning foundational skills, such as language understanding or image recognition, foundation models can be adapted quickly to specific tasks or industries.

Some  examples include:
- **BERT/GPT**: Popular foundation models in natural language processing, enabling applications ranging from sentiment analysis to text generation.
- **CLIP/DALLE**: Models combining text and image modalities, showcasing robust performance in both vision and linguistic tasks.
- **ConvNets trained on ImageNet**: these models are easy to use for downstream tasks by fine-tuning and have led to all of the advances in applied computer vision in the last decade.