---
title: "Semantic segmentation"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Image segmentation is a fundamental task in computer vision, aiming to classify each pixel of an image to identify structures or objects. Unlike object detection, which provides bounding boxes for objects, image segmentation requires finer granularity by labeling each pixel with its belonging class.

## Standard architecture: UNET

UNET ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) was originally specifically designed for biomedical image segmentation, but it is an established standard for semantic segmentation in general. It addresses tasks where precise localization is crucial by transforming input images into pixel-level masks efficiently.

Key features are:
- **Symmetrical Architecture**: The UNET employs a U-shaped structure comprising both contracting and expanding paths, allowing it to precisely capture context while maintaining spatial detail.
- **Encoder (Contracting Path)**: This path involves multiple convolutional layers followed by pooling operations, progressively downsampling the feature maps to extract high-level spatial information.
- **Decoder (Expanding Path)**: Through upsampling using transposed convolutions, this path reconstructs the image dimensions. Skip connections from the encoder ensure detailed spatial recovery.
- **Skip Connections**: These connections bridge corresponding layers of encoder and decoder, combining coarse semantic knowledge with fine-grained features. This enables robust, pixel-accurate segmentations by preserving original image data.

For each pixel, the model predicts the probability of belonging to a specific class.

### Training and related loss functions

Training the UNET model involves minimizing a loss function that captures both class membership accuracy and spatial coherence within predicted masks.

1. **Cross Entropy Loss**:
   - Measures classification errors at the pixel level, treating each pixel as an independent class prediction.
   - Effective for multi-class segmentation but may require adaptation for imbalanced classes.

2. **Dice Coefficient Loss**:
   - Evaluates overlap between predicted and ground truth masks.
   - Particularly useful for medical images where correctly predicting boundaries is critical.

Combinations of these losses can be employed, with weights adjusted based on dataset characteristics and task objectives.
Hyperparameters such as learning rate and batch size significantly influence convergence behavior and final model performance.

## Alternatives to the UNET architecture

While UNET remains popular for its simplicity and effectiveness, alternative architectures have emerged offering potential improvements:

1. **Fully Convolutional Networks (FCNs)**:
   - **Approach**: Extends traditional CNNs to enable dense predictions on full-sized images.
   - **Strengths**: Provides flexibility across various image sizes and establishes a foundation for modern segmentation tasks.

2. **DeepLab Models**:
   - **Approach**: Introduces atrous convolutions to expand receptive fields without increasing parameter counts.
   - **Variants**: Includes DeepLabV3, which incorporates improved spatial pyramid pooling mechanisms.
   - **Strengths**: Excel in balancing scale variance and segmentation precision across complex scenes.



## Code Example: Training a UNET for Image Segmentation

Below is a simple example of how to implement and train a UNET model for image segmentation using TensorFlow and Keras:

Import Required Libraries:
```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import concatenate, Input, Dropout
from tensorflow.keras.models import Model
```

Build the UNET Model:
```python
def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    
    # Decoder
    up4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
    merge4 = concatenate([up4, conv2], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
    merge5 = concatenate([up5, conv1], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
```

Compile and train the model

```python
# Create an instance of the model
model = unet_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dummy data for illustration
import numpy as np
X_train = np.random.rand(10, 128, 128, 3)
y_train = np.random.randint(0, 2, (10, 128, 128, 1))

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=2)
```

This code snippet constructs a basic UNET architecture suitable for binary image segmentation tasks. Note that actual datasets should replace the dummy data, and hyperparameters might need tuning based on specific use cases.
Feel free to tweak and expand this code for more complex datasets and training regimes!

## Further resources

1. Comprehensive overview on UNET architecture: [Uni Freiburg / Olaf Ronneberger](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
2. Advanced image segmentation tutorial using PyTorch: [Pytorch Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)