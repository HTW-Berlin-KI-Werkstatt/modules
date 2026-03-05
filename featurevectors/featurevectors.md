---
title: "Feature vectors and embeddings"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

As we have seen in the previous examples, classical machine learning models require vectors as inputs. These vectors are
often refered to as *feature vectors* or *embeddings* or in more general *representations*. We will come to the subtle differences later on.
But how do we turn our inputs (text, image, video, etc.) into vectors? 
What would be good representations?
Let's answer the last question first.

## What is a good representation?

This question is best answered for a classification tasks and the goal is always to have a representation that is very
easy for the classication model to learn from:
1. **Discriminative Power:** The ability to distinguish between different classes effectively.
2. **Invariance**: Stability to transformations such as rotation, scaling, or translation in the input data.
3. **Noise Robustness:** The ability to handle and perform well even when the data contains noise or irrelevant features.
4. **Relevant Features:** Focusing on relevant features.

Some traditional machine learning scientist would still mention *low dimensionality* here, but this is open to debate.

## Feature extraction

Deriving good features by manually designed algorithms has been the most important task of a machine learning engineer back in the days and for certain
applications it still is. Figuring out how to turn high-dimensional images into compact and discriminate features for example highly depends on the application - is it
important to consider color or is it rather important to gather statistics of shapes in the image?
Let's look into an example from the computer vision (CV) and the natural language processing domain (NLP):

### CV example: Extracting Color Feature Vectors from Images

Let's start with a simple example: extracting color histograms as feature vectors using OpenCV.

```python
import cv2
import numpy as np

# Function to convert image to color histogram feature vector
def color_histogram_feature_vector(image, bins=(8, 8, 8)):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute the color histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Load an example image
image = cv2.imread('example.jpg')

# Convert the image to a color histogram feature vector
feature_vector = color_histogram_feature_vector(image)

print(f"Feature vector shape: {feature_vector.shape}")
print(f"Feature vector: {feature_vector[:10]}...")  # Print first 10 elements for brevity
```

In this example, the ``color_histogram_feature_vector`` function converts an image into the HSV (hue, saturation, value) color space and computes a normalized color histogram. The histogram is then flattened into a feature vector. This could be a suitable feature vector for distinguishing between red and blue products placed on a manufacturing line.

### NLP Example: Bag of Words (BoW)

The **Bag of Words (BoW)** model represents text by counting the frequency of each word within a document. This method ignores grammar and word order but considers the multiplicity of words.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Create the Bag of Words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Convert sparse matrix to array
feature_vectors = X.toarray()

print(f"Feature vectors shape: {feature_vectors.shape}")
print(f"Vocabulary: {vectorizer.get_feature_names_out()}")
print(f"Feature vectors:\n{feature_vectors}")
```

In this example, the `CountVectorizer` from scikit-learn does the whole job. 
The result is a matrix where each row corresponds to a document and each column corresponds to a word's frequency in that document.

## Learned Embeddings

Designing feature vectors is an enourmous challenge, but how about we also leave this job 
to the machine itself and learn how to extract features. When we learn how to extract features from data, the term *embedding* is more
common to be used. We will learn later on how these embeddings are learned, let's rather jump into some examples showing how to use them.

### NLP Example: Using CLIP Model to Obtain Text Embeddings

For this example, we'll use the CLIP model of Open AI [(Radford et al., 2021)](https://arxiv.org/abs/2103.00020) to obtain text embeddings. The embedding model consists of several parameters, which
we obtain from [huggingface.co](https://huggingface.co) - an amazing open portal consisting of models, datasets, and benchmarks.
Their python module ``transformers`` works like a charm and takes care of the download and caching in the background:

```python
import torch
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to get the text embedding using CLIP
def get_text_embedding(text):
    # Preprocess the text
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)

    # Generate the embedding
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

    # Return the embedding as a numpy array
    return outputs[0].cpu().numpy().flatten()

# Getting the embedding for an example text
example_text = "This is an example sentence."
embedding_vector = get_text_embedding(example_text)

print(f"Embedding vector shape: {embedding_vector.shape}")
print(f"Embedding vector: {embedding_vector[:10]}...")  # Print first 10 elements for brevity
```

The resulting embedding is a high-dimensional vector representing related to the semantic meaning of the text.

### CV Example: Using CLIP Model to Obtain Text Embeddings

The most important aspects of CLIP is that it offers computing image embeddings in addition and we just have to 
add the following function:

```python
def get_image_embedding(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    # Generate the embedding
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Return the embedding as a numpy array
    return outputs[0].cpu().numpy().flatten()
```

This is even not the end of the story: in CLIP, text and image share the same embedding space. If an image $$I$$ is semantically close to a text $$T$$,
their embeddings will have a close distance. This offers an enourmous potential for fast algorithm development, something that you might have even used beforehand.

## Further reading

1. Embedding for audio signals: Wav2Vec [(Baevski et al, 2020)](https://arxiv.org/abs/2006.11477)
2. One of the first papers on word embeddings: Word2Vec [(Mikolov et al, 2013)](https://arxiv.org/pdf/1301.3781)
3. Great tool for exploring embeddings: [Embedding Projector](https://projector.tensorflow.org/)
