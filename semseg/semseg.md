---
title: "Semantic segmentation"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

Image segmentation is a fundamental task in computer vision, aiming to classify each pixel of an image to identify structures or objects. Unlike object detection, which provides bounding boxes for objects, image segmentation requires finer granularity by labeling each pixel with its belonging class.

## Standard architecture: U-Net

U-Net ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) was originally specifically designed for biomedical image segmentation, but it is an established standard for semantic segmentation in general. It addresses tasks where precise localization is crucial by transforming input images into pixel-level masks efficiently.

Key features are:
- **Symmetrical Architecture**: The U-Net employs a U-shaped structure comprising both contracting and expanding paths, allowing it to precisely capture context while maintaining spatial detail.
- **Encoder (Contracting Path)**: This path involves multiple convolutional layers followed by pooling operations, progressively downsampling the feature maps to extract high-level spatial information.
- **Decoder (Expanding Path)**: Through upsampling using transposed convolutions, this path reconstructs the image dimensions. Skip connections from the encoder ensure detailed spatial recovery.
- **Skip Connections**: These connections bridge corresponding layers of encoder and decoder, combining coarse semantic knowledge with fine-grained features. This enables robust, pixel-accurate segmentations by preserving original image data.

For each pixel, the model predicts the probability of belonging to a specific class.

### Training and related loss functions

Training the U-Net model involves minimizing a loss function that captures both class membership accuracy and spatial coherence within predicted masks.

1. **Cross Entropy Loss**:
   - Measures classification errors at the pixel level, treating each pixel as an independent class prediction.
   - Effective for multi-class segmentation but may require adaptation for imbalanced classes.

2. **Dice Coefficient Loss**:
   - Evaluates overlap between predicted and ground truth masks.
   - Particularly useful for medical images where correctly predicting boundaries is critical.

Combinations of these losses can be employed, with weights adjusted based on dataset characteristics and task objectives.
Hyperparameters such as learning rate and batch size significantly influence convergence behavior and final model performance.

## Alternatives to the U-Net architecture

While U-Net remains popular for its simplicity and effectiveness, alternative architectures have emerged offering potential improvements:

1. **Fully Convolutional Networks (FCNs)**:
   - **Approach**: Extends traditional CNNs to enable dense predictions on full-sized images.
   - **Strengths**: Provides flexibility across various image sizes and establishes a foundation for modern segmentation tasks.

2. **DeepLab Models**:
   - **Approach**: Introduces atrous convolutions to expand receptive fields without increasing parameter counts.
   - **Variants**: Includes DeepLabV3, which incorporates improved spatial pyramid pooling mechanisms.
   - **Strengths**: Excel in balancing scale variance and segmentation precision across complex scenes.



## Code Example: Training a U-Net for Image Segmentation

Below is a simple example of how to implement and train a small U-Net model for binary image segmentation using PyTorch:

```python
import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        # Encoder (contracting path)
        self.conv1 = double_conv(in_channels, 64)
        self.conv2 = double_conv(64, 128)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.conv3 = double_conv(128, 256)
        # Decoder (expanding path)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv4 = double_conv(256, 128)   # 128 from up4 + 128 from the skip connection
        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5 = double_conv(128, 64)    # 64 from up5 + 64 from the skip connection
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.pool(c1))
        c3 = self.conv3(self.pool(c2))
        u4 = self.conv4(torch.cat([self.up4(c3), c2], dim=1))  # skip connection
        u5 = self.conv5(torch.cat([self.up5(u4), c1], dim=1))  # skip connection
        return self.out(u5)  # logits, one channel per class
```

Train the model on dummy data:

```python
model = UNet()
criterion = nn.BCEWithLogitsLoss()  # sigmoid + binary cross entropy on the pixel level
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy data for illustration: 10 RGB images of size 128x128 with binary masks
X_train = torch.rand(10, 3, 128, 128)
y_train = torch.randint(0, 2, (10, 1, 128, 128)).float()

for epoch in range(5):
    for i in range(0, len(X_train), 2):  # batch size 2
        optimizer.zero_grad()
        logits = model(X_train[i:i+2])
        loss = criterion(logits, y_train[i:i+2])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

This code snippet constructs a basic U-Net architecture suitable for binary image segmentation tasks. Note that actual datasets should replace the dummy data, and hyperparameters might need tuning based on specific use cases.
Feel free to tweak and expand this code for more complex datasets and training regimes!

## Further resources

1. Comprehensive overview on U-Net architecture: [Uni Freiburg / Olaf Ronneberger](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
2. Instance segmentation (Mask R-CNN) tutorial using PyTorch: [Pytorch Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)