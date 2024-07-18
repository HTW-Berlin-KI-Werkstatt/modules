---
title: "Math essentials"
layout: single
author_profile: true
author: Erik Rodner
lecture_name: "Machine Learning"
lecture_desc: "From linear models to GPTs"
licence: "CC-BY"
licence_desc: 2024 | HTW Berlin 
toc: false
classes: wide
---

## Do we really need math to do machine learning projects?

The short answer is yes. While there are many tools and libraries that abstract away much of the complexity, a solid understanding of the underlying mathematics is crucial for several reasons:

1. **Model Selection:** Understanding the mathematics behind different models helps in selecting the appropriate model for your specific problem.
2. **Hyperparameter Tuning:** Many machine learning models have hyperparameters that need to be fine-tuned. Knowledge of math can guide the tuning process.
3. **Understanding Results:** Interpreting the results of a machine learning model requires a good grasp of statistics and probability.
4. **Troubleshooting:** When models do not perform as expected, mathematical insights can help diagnose and fix issues.

Therefore, while it is possible to get started with machine learning using high-level libraries and frameworks, mastering the mathematical foundations is necessary for advanced projects and innovation, something which should be the goal of each master student.

## Math aspects you need to survive

To effectively work on machine learning projects, you need to be familiar with several key mathematical concepts:

- **Vectors and Matrices:** Data is represented as vectors and matrices and therefore are a crucial element in ML (Resources: [MML Book, Sect. 2.1 & Sect. 2.2](https://mml-book.github.io/book/mml-book.pdf), [Khan academy](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices) [⭐Intuitive view on linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)) 
  1. Remind yourself what matrix/vector multiplications are and how to calculate them! ([Online Test at Khan Academy](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/v/matrix-multiplication-intro))
  2. What is a transpose and an inverse of a matrix?
  3. What is a linear equation system and how to solve it?
- **Differentiation and Integration:** Fundamental for optimization problems (Resources: [⭐Geometric view on calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr))
  1. What are the basic differentiation rules that you remember from school?
- **Distributions and probabilities:** Understanding different types of data distributions. (Resources: [Statistics for Data Science without Hypothesis Testing](https://chkra.github.io/ds-lecture/modules/statistical-inference/stat-inf/))
  1. You need to understand what a normal distribution is and the effect of the mean and standard deviations.
  2. What is a conditional distribution?
  3. What is an expected value, what is the standard deviation?

Having a good grasp of these areas will help you dramatically to explore the machine learning world without relying on ease-of-use software and without missing new advances. We will practice quite a lot of these things in lecture.

I skipped on purpose a few topics that would be important if we had more time like eigenvectors and hypothesis testing. The latter one was discussed in the data science lecture of Christina Kratsch and is not essential for understanding the next parts of this lecture.

## Test yourself

1. Let $$\mathbf{A} = \begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix}$$, write down $$\mathbf{A}^T$$ and calculate $$\mathbf{A}^{-1}$$!
2. If $$\mathbf{x} = \begin{bmatrix} 1 \\\ -1 \end{bmatrix}$$, calculate
  * $$\mathbf{A} \cdot \mathbf{x}$$, 
  * $$\mathbf{x}^T \cdot \mathbf{x}$$, and 
  * $$\mathbf{A} \mathbf{A}^T$$.
3. Let $$f = x^2 y + (2y - 1)^2$$, compute the derivative of 
  * $$f$$ with respect to $$x$$ and 
  * $$f$$ with respect to $$y$$. 
4. Let $$y$$ be a random variable representing the height of a person and $$g$$ being a discrete random variable representing the gender (assumed to be binary), what does the term $$p (y  \;\vert\; g = \text{male} )$$ refer to? What does $$p(g = \text{female})$$ refer to?