---
title: "Object detection - architecture and losses"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Object detection is a crucial computer vision task that involves identifying and localizing objects within an image. It extends beyond simple classification by detecting multiple objects simultaneously and providing their *bounding box*.

## Standard architecture: YOLO (You Only Look Once)

YOLO ([Redmon, Divvala, Girshick, Farhadi, 2015](https://arxiv.org/abs/1506.02640)) is an object detection model notable for its real-time speed and accuracy. Unlike traditional methods, which repurpose classifiers or localizers to perform detection, YOLO frames object detection as a single regression problem, directly predicting bounding boxes and class probabilities from full images in one evaluation. 

Key features are:
- **Unified Architecture**: YOLO processes the *entire image with a single convolutional network*, predicting bounding boxes and class probabilities simultaneously.
- **Speed**: Due to its real-time processing capability, YOLO can achieve high frame rates, making it highly suitable for applications requiring rapid responses, such as autonomous driving.
- **Global Context**: By considering global image context during detection, YOLO reduces false positives, particularly where similar objects are close together.
- **Grid Division**: The image is divided into an $$ S \times S $$ grid, with each cell responsible for predicting bounding boxes if the center of an object falls within the cell. Each grid can predict $$ B $$ bounding boxes in parallel.

For each bounding box, the model predicts:
  - Coordinates (x, y, width, height)
  - Confidence score indicating probability that a box contains an object and effectiveness of the box
  - Class probabilities for each object class

What is the maximum number of objects, a YOLO model can detect?

### Training and related loss functions

When such a model is trained, different goals apply, since you want the model to avoid class confusion (detecting a motorbike instead of a car), be precise with the localizations (do not predict a too wide or too narrow box) and be robust (avoid false positives). 
Each goal is formulated as an individual loss function and during training a weighted sum of these functions
is minimized.

1. **Localization Loss**:
   - Measures errors in predicted bounding box coordinates and sizes (square root transform).

2. **Confidence Loss**:
   - Quantifies how accurately the presence of an object in a bounding box is predicted.
   - Involves two components:
     - Objectness score when there's an object (IOU between predicted and ground truth boxes).
     - Score when there isn't an object, driving down false positives.

3. **Classification Loss**:
   - Evaluates the accuracy of predicted class probabilities within each bounding box.


YOLO combines these components into a weighted sum to form the total loss function.

Hyperparameters like $$\lambda_{\text{coord}}$$ and $$\lambda_{\text{class}}$$ are used to balance different parts of the detection task according to importance.

Over time, several variants have been developed to further improve YOLO's performance.

## Alternatives to the YOLO architecture

While YOLO is prominent due to its speed, other object detection models provide valuable alternatives:

1. **R-CNN (Region-based Convolutional Neural Networks)**:
   - **Approach**: A two-stage detector involving region proposals followed by classification.
   - **Variants**: Includes Fast R-CNN and Faster R-CNN, improving speed and accuracy by streamlining proposal generation and using neural networks end-to-end.
   - **Strengths**: High precision but typically slower than YOLO.

2. **SSD (Single Shot MultiBox Detector)**:
   - **Approach**: Involves a single forward pass through the network, similar to YOLO, predicting category scores and box offsets directly.
   - **Architecture**: Utilizes feature maps of different resolutions to handle varying object sizes, enhancing detection robustness.
   - **Strengths**: Balances trade-offs between speed and accuracy, often outperforming YOLO on certain tasks.

## Further ressources

1. Further details explained: [](https://www.datacamp.com/blog/yolo-object-detection-explained)
2. Ultralytics object detection: [](https://docs.ultralytics.com/de/tasks/detect/)
3. Yet another blog article on YOLO v1: [](https://towardsdatascience.com/evolution-of-yolo-yolo-version-1-afb8af302bd2)