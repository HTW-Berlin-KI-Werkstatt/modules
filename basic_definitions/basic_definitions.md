---
title: "Basic definitions"
layout: single
author_profile: true
author: Erik Rodner
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
toc: false
classes: wide
---

## What is machine learning?

**Machine learning (ML)** is often seen as a subset of **artificial intelligence (AI)** that focuses on the development of algorithms and statistical models that enable computers to perform specific tasks without using explicit instructions. Instead, these systems build a model based on *training* data to make predictions or decisions without being programmed to perform the task.

The current advances contributed to the field of AI are all based on ML. Therefore, people often mix these terms. This is critical, since AI often suggests something magical close to the human brain, which is not the case.

A further sub-part of machine learning is **deep learning**. However, the difference between classical machine learning and deep learning is even more fuzzy and we come to this later.

Advice: use the term machine learning for technical communications (talking about algorithms, best practices, explaining your software in detail) and only use the term AI for non-technical communications (getting venture capital, talking about company strategies, etc.).
{: .notice--primary}

## Sub-fields of machine learning


Machine learning is broadly categorized into three types:
- **Supervised Learning:** The algorithm learns from labeled training data, i.e. data that not only includes inputs but also expected outputs (often referred to as labels or annotations) provided by human annotator. Nearly all of the ML applications nowadays rely on supervised learning.
- **Unsupervised Learning:** The algorithm is given data without provided labels. The model needs to infer the tasks from the data structure alone. Unsupervised learning is often used in a early phase to detect interesting aspects of the data or help the annotation process.
- **Reinforcement Learning:** The algorithm learns by interacting with its environment and receiving rewards for performing actions that lead to positive outcomes.

Exercise: What are applications you know that involve machine learning? What kind of training data might it need?
{: .notice--info}