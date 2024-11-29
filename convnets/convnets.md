---
title: "Convolutional neural networks"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Convolutional Neural Networks have revolutionized the field of computer vision by enabling machines to understand visual data. This lecture will unpack how CNNs operate, focusing on their unique components and the motivation behind their design. Please also visit [CNN-Explainer](https://poloclub.github.io/cnn-explainer/), which is a really great interactive tool that visualizes a CNN architecture and the computations therein.

## Input Representation as Tensors

In a CNN, images are represented as tensors. A tensor is essentially a multi-dimensional array that can include scalars, vectors, matrices, and higher-dimensional counterparts.

Since we are dealing with images, machine learning models for computer vision that
take the image directly without computing features beforehand, require the following inputs:
- Grayscale images are represented as tensors of shape $$(H, W, 1)$$, where $$ H $$ is height 
and $$ W $$ is width.
- RGB images appear as $$(H, W, 3)$$, reflecting three color channels (Red, Green, Blue).

## Convolutions

Convolutions are the heart of CNNs, designed to automatically detect patterns in input data using filters, known also as kernels.

A convolution operation involves sliding a kernel across input data to produce a feature map. If you imagine a kernel as a small matrix, it captures patterns such as edges or textures by dot multiplying its weights with overlapping input regions.

![](img/no_padding_no_strides.gif)
> Convolution operation without padding and strides [^1]

[^1]: Animations from Vincent Dumoulin, Francesco Visin - *A guide to convolution arithmetic for deep learning* [pdf](https://arxiv.org/abs/1603.07285)

Let's make an example of how a convolution works, the input is given as $$5 \times 5$$ matrix:
```
[1, 2, 3, 0, 1]
[4, 5, 6, 0, 0]
[7, 8, 9, 1, 2]
[0, 1, 2, 3, 4]
[1, 1, 0, 0, 1]
```

A convolution uses a mask or filter given as a matrix (here $$3 \times 3$$):
```
[1, 0, -1]
[1, 0, -1]
[1, 0, -1]
```

Now imagine you slide the mask across the input image and at each step you multiple the mask
with the respective region of the image and sum up all the values. The sum is then the new value of the center pixel. All of these values lead to a new matrix, which is the result of the convolution.

1. **Top-left position:**  
   $$(1 \cdot 1) + (2 \cdot 0) + (3 \cdot -1) + (4 \cdot 1) + (5 \cdot 0) + (6 \cdot -1) + (7 \cdot 1) + (8 \cdot 0) + (9 \cdot -1) =$$ $$1 - 3 + 4 - 6 + 7 - 9 = -6$$

2. **Next position to the right:**  
   $$(2 \cdot 1) + (3 \cdot 0) + (0 \cdot -1) + (5 \cdot 1) + (6 \cdot 0) + (0 \cdot -1) + (8 \cdot 1) + (9 \cdot 0) + (1 \cdot -1) =$$ $$2 + 5 + 8 - 1 = 14$$

... and so on. We get the resulting matrix (often called *features map*):

```
[-6, 14, 8]
[ 3, 12, 10]
[-4, 2, 6]
```

Please note that we only computed values for pixels of the input image that are not on the border of the image, i.e. which allow the filter matrix to be centered on.
Convolution operations in neural networks have of course again several parameters to play with:

- **Filter Size** controls the size of the filter in lateral directions. It has been shown by plenty of empirical evaluations that it is reasonable to use small filter sizes ($$3 \times 3$$ or $$5 \times 5$$) but a lot of layers.
- **Stride** controls the step size when moving the filter. A stride of 1 means shifting the filter one pixel at a time.
- **Padding** affects output size. Adding zeros around the input perimeter allows the filter to fit better. "Same" padding maintains output size, while "valid" padding reduces it.

![](img/padding_strides.gif)
> Convolution operation with 1 pixel padding and a stride of 2 pixels [^1]

## Activation Functions

After applying convolutions, the activation function introduces non-linearity. ReLU (Rectified Linear Unit) is commonly used, defined as \( A(x) = \max(0, x) \). It replaces negative values with zero and retains positive values, allowing networks to handle complex data patterns effectively.

## Pooling Layers

Pooling layers reduce dimensionality, increasing computational efficiency and robustness to position variations.

The most common form, max pooling, involves taking the maximum value from patches within the feature map. For instance, given the following $$2 \times 2$$ patch:
```
Patch:
[1, 3]
[2, 4]

Max Pooling Result: 4
```

This process retains salient features while reducing spatial dimensions.

## Designing architectures

After several convolutional and pooling operations, CNNs transition to fully connected layers, similar to traditional neural networks. The purpose of these layers is to integrate the high-level features detected by convolutions across the image into a decision-making process for classification tasks.

Various architectures have been developed to optimize CNN performance for specific tasks:

1. **LeNet-5**: One of the earliest CNN architectures, designed for handwritten digit recognition. It consists of two sets of convolutional and pooling layers, followed by fully connected layers.

2. **AlexNet**: A deep CNN architecture famous for winning the ImageNet competition in 2012. It introduced the use of ReLU non-linearities and dropout layers to mitigate overfitting, alongside multiple convolutional and fully connected layers.

3. **VGGNet**: Known for its simplicity and uniform architecture, VGGNet uses very small (3x3) convolution filters and emphasizes depth with up to 19 layers, which helps capture intricate patterns.

4. **GoogLeNet (Inception)**: This architecture uses "Inception modules" which allow the network to consider different filter sizes within a single layer, improving both feature extraction and computational efficiency.

5. **ResNet (Residual Networks)**: Introduces skip connections, or residual blocks, which solve the vanishing gradient problem, allowing training of very deep networks (e.g., hundreds of layers).

## Classification Using CNNs

After several convolutional and pooling operations, CNNs transition to fully connected layers, similar to traditional neural networks.

A common loss function for classification is Softmax Cross-Entropy. In this setting, the softmax function converts logits (raw model outputs) into probabilities, and cross-entropy measures the difference between predicted probabilities and true labels. We have seen this concept already in a previous lecture.

