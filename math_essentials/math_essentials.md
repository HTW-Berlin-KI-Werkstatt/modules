---
title: "Math essentials"
layout: single
author_profile: true
author: Erik Rodner
toc: false
classes: wide
---

## Do we really need math to do machine learning projects?

Yes, although maybe not in the way you fear. Libraries like scikit-learn and PyTorch hide most of the computations, and a first model is trained in five lines of code. The math becomes necessary the moment something does not work as expected or a number has to be interpreted. Three situations you will run into in this course:

Your PyTorch model stops with `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x784 and 512x10)`. If you know that a $$(m \times n)$$ matrix can only be multiplied with an $$(n \times k)$$ matrix, you find the wrong layer within a minute instead of by trial and error.

You search for similar documents with text embeddings and have to choose between Euclidean and cosine distance. Which one fits depends on whether the length of an embedding vector carries meaning, and that is a question about dot products and norms.

A classifier detects 90% of all sick patients and raises a false alarm for only 10% of the healthy ones. Still, only about 8% of its positive predictions are correct (see the binary test example below). Without conditional probabilities this looks like a contradiction; with Bayes' rule it is a two-line calculation, and it shows why a single accuracy number is rarely enough to judge a model.

Getting started with high-level libraries works without much math. Understanding why a model fails, reading a new paper or designing your own method does not, and the latter should be the goal of each master student.

## Math aspects you need to survive

To effectively work on machine learning projects, you need to be familiar with two key areas: linear algebra and probability. Both are collected below. Pick up the notation as you go, and ask whenever a symbol is unclear. Exercises to test yourself are at the end of the page.

Having a good grasp of these areas will help you dramatically to explore the machine learning world without relying on ease-of-use software and without missing new advances. We will practice quite a lot of these things in lecture.

I skipped on purpose a few topics that would be important if we had more time, like calculus, eigenvectors and hypothesis testing. For differentiation, the basic rules you remember from school are enough for now; see [Further reading](#further-reading) for a geometric view on calculus. [Hypothesis testing](/modules/statistical-inference/stat-inf.md#hypothesis-testing) was discussed in the data science lecture of Christina Kratsch and is not essential for understanding the next parts of this lecture.

### Linear algebra

Data in ML is usually stored as a matrix with one example per row. Almost every model step is a dot product or a matrix–vector product, so you should be able to compute these by hand and check shapes in your head.

**Dot product.** Multiply two vectors entry by entry and sum up the products. The dot product is the core of linear models, attention and cosine similarity; below we will see that it measures how aligned two vectors are.

$$
\mathbf{w}^T \mathbf{x} = \sum_{d=1}^{D} w_d \, x_d
$$

**Norms and distances.** The Euclidean norm is the length of a vector and follows directly from the dot product; the distance of two vectors is the length of their difference. These are used, for example, by nearest neighbours ([lecture 03](/lectures/03/03.md)):

$$
\lVert\mathbf{x}\rVert_2 = \sqrt{\mathbf{x}^T \mathbf{x}} \qquad d(\mathbf{x}, \mathbf{z}) = \lVert\mathbf{x} - \mathbf{z}\rVert_2
$$

**Angle and dot product.** The dot product, the lengths and the angle $$\theta$$ between two vectors are connected:

$$
\mathbf{x}^T \mathbf{z} = \lVert\mathbf{x}\rVert_2 \, \lVert\mathbf{z}\rVert_2 \cos \theta
$$

Dividing the dot product by both lengths therefore leaves only the angle: $$\cos \theta = 1$$ for the same direction, $$0$$ at a right angle and $$-1$$ for opposite directions.
The angle ignores the lengths and compares only directions. Turning it into a distance, which is 0 for the same direction and grows as the directions differ, gives the cosine distance. It is commonly used for text and embeddings, while the Euclidean distance remains the default when lengths carry meaning.

$$
\cos \theta = \frac{\mathbf{x}^T \mathbf{z}}{\lVert\mathbf{x}\rVert_2 \, \lVert\mathbf{z}\rVert_2} \qquad d_{\cos}(\mathbf{x}, \mathbf{z}) = 1 - \cos \theta \qquad d_{\mathrm{euc}}(\mathbf{x}, \mathbf{z}) = \lVert\mathbf{x} - \mathbf{z}\rVert_2
$$

![Example vectors x = (3, 1) and z = (1, 2): angle, cosine distance and Euclidean distance](/modules/math_essentials/images/angle_vs_distance.svg)

Doubling $$\mathbf{z}$$ would leave the angle and the cosine distance unchanged but move its tip, so the Euclidean distance would change.

**Matrix–vector product.** Multiplying a vector $$\mathbf{x}$$ with a matrix $$\mathbf{A}$$ is the simplest way to transform data represented as vectors: an $$(m \times n)$$ matrix maps an $$n$$-dimensional vector to an $$m$$-dimensional one. We will see many algorithms that apply such transformations many times in sequence to achieve a certain result. Each output entry is the dot product of one row of $$\mathbf{A}$$ with $$\mathbf{x}$$.

$$
(\mathbf{A}\mathbf{x})_i = \sum_j A_{ij} \, x_j
$$

**Matrix–matrix product and Falk's scheme.** Multiplying $$\mathbf{A}$$ with a matrix $$\mathbf{B}$$ applies the matrix–vector product to every column of $$\mathbf{B}$$. Falk's scheme is a good way to keep in mind how this works and which shapes are needed: write $$\mathbf{B}$$ above and to the right of $$\mathbf{A}$$. Each entry of $$\mathbf{A}\mathbf{B}$$ is the dot product of the row of $$\mathbf{A}$$ and the column of $$\mathbf{B}$$ that cross at it, and the layout checks shapes: the columns of $$\mathbf{A}$$ must line up with the rows of $$\mathbf{B}$$.

![Falk's scheme for A (2×3) times B (3×2)](/modules/math_essentials/images/falk_scheme.svg)

**Shape rule.** $$(m \times n)$$ times $$(n \times k)$$ gives $$(m \times k)$$, and the inner sizes must match. Write the shapes down before every product. This is the same check as debugging a PyTorch shape error.

**Matrix product is not commutative.** In general $$\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$$, and often only one of the two is even defined.

**Transpose and inverse.** These rules are needed to read the least-squares solution in [lecture 05](/lectures/05/05.md). A matrix is invertible only if it is square and its determinant is not 0.

$$
(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T \qquad \begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

**Linear equation systems.** A system $$\mathbf{A}\mathbf{x} = \mathbf{b}$$ with a square, invertible $$\mathbf{A}$$ has the unique solution $$\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$$. In code, use `np.linalg.solve(A, b)` instead of computing the inverse explicitly; it is faster and numerically more stable.

### Probability and statistics

Classifiers output probabilities, and their usual loss, cross-entropy, is a negative log-probability. You need conditional probability, Bayes' rule and the Gaussian to read [lecture 05](/lectures/05/05.md), [lecture 08](/lectures/08/08.md) and [lecture 09](/lectures/09/09.md).

**Random variable and distribution.** A random variable maps outcomes to values. A discrete one has probabilities $$p(x)$$ that sum to 1.

**Running example: a binary test.** A medical test, or an ML classifier, maps each patient to $$T = 1$$ (positive) or $$T = 0$$ (negative). A second random variable $$K$$ says whether the patient has the disease (1) or not (0). Say 1% of people are sick, the test detects 90% of sick people, and it is also positive for 10% of healthy people. Before reading on, make a guess: someone tests positive, how likely are they to be sick?

**Counting 1,000 people.** Imagine 1,000 people and sort them by $$K$$ and $$T$$ into a 2×2 table. The example tells us directly that 10 of them are sick and 990 are healthy. Everything else is still open.

|  | T = 1 (positive) | T = 0 (negative) | sum |
| --- | --- | --- | --- |
| K = 1 (sick) | ? | ? | 10 |
| K = 0 (healthy) | ? | ? | 990 |
| sum | ? | ? | 1,000 |

**Filling the cells.** How can we calculate the missing values in the table? Of the 10 sick people, 90% (as given above), i.e. 9 people, test positive, and the remaining 1 tests negative. The other 990 are healthy: 10% of them, i.e. 99 people, test positive (as given above), and the remaining 891 test negative.

|  | T = 1 (positive) | T = 0 (negative) | sum |
| --- | --- | --- | --- |
| K = 1 (sick) | 9 | 1 | 10 |
| K = 0 (healthy) | 99 | 891 | 990 |
| sum | 108 | 892 | 1,000 |

**Joint distribution.** Dividing every count by 1,000 turns the table into the joint distribution $$p(K, T)$$. It gives one probability per combination, and its four cells sum to 1.

| p(K, T) | T = 1 (positive) | T = 0 (negative) | sum |
| --- | --- | --- | --- |
| K = 1 (sick) | 0.009 | 0.001 | 0.01 |
| K = 0 (healthy) | 0.099 | 0.891 | 0.99 |
| sum | 0.108 | 0.892 | 1 |

**Marginalization.** Summing out one variable gives the distribution of the other: the column sums are $$p(T)$$ and the row sums are $$p(K)$$. For example, $$p(T = 1) = 0.009 + 0.099 = 0.108$$.

$$
p(t) = \sum_{k} p(k, t) \qquad p(k) = \sum_{t} p(k, t)
$$

**Conditional probability.** The joint table answers questions about all 1,000 people, such as how many are sick and test positive. Often we already know something and only care about a group. Back to the original question: someone tests positive, how likely are they to be sick? The joint table does not answer it directly, but we can use it to compute so-called conditional probabilities. The probability of $$a$$ given $$b$$ is defined as:

$$
p(a \mid b) = \frac{p(a, b)}{p(b)}
$$

Let's apply this directly to answer the question:

$$
p(K{=}1 \mid T{=}1) = \frac{p(K{=}1, T{=}1)}{p(T{=}1)} = \frac{0.009}{0.108} \approx 0.083
$$

That is about an 8% chance of being sick, or 9 of the 108 people who test positive.

**Bayes' rule.** Our step-by-step calculation fits into one formula, Bayes' rule. The numerator $$p(T{=}1 \mid K{=}1)\, p(K{=}1) = 0.9 \cdot 0.01 = 0.009$$ is the cell we filled by counting. The denominator $$p(T{=}1) = 0.108$$ is the column sum from marginalization.

$$
p(K \mid T) = \frac{p(T \mid K)\,p(K)}{p(T)} \qquad p(K{=}1 \mid T{=}1) = \frac{0.9 \cdot 0.01}{0.108} \approx 0.083
$$

**Continuous random variables.** $$K$$ and $$T$$ take only two values, so a table holds their whole distribution. Many quantities in ML are measured rather than counted, such as a height, a temperature, an arrival time or a model's output score. They can take any value in a range, so a table is impossible, and the chance of hitting one exact value, say an arrival time of exactly 10:00:00.000, is 0. Instead we describe them by a density $$p(x)$$. The probability of landing in an interval is the area under $$p(x)$$ over that interval, and the total area is 1.

$$
P(a \le x \le b) = \int_a^b p(x)\,dx \qquad \int_{-\infty}^{\infty} p(x)\,dx = 1
$$

**Example task: when to close the door?** Students arrive at a lecture at slightly different times, on average at 10:00, give or take about 3 minutes. Arrival time is a continuous random variable. When can the lecturer close the door so that almost everyone is already inside? To answer this we need a density for the arrival times.

**Gaussian.** The Gaussian is the most common density. Its values can exceed 1, because only areas under it are probabilities. The mean $$\mu$$ shifts the curve and the standard deviation $$\sigma$$ sets its width.

$$
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

Let's model the arrival times as Gaussian with $$\mu = $$ 10:00 and $$\sigma = 3$$ min. For a Gaussian we can use the 68–95–99.7 rule: about 68% arrive within $$\mu \pm \sigma$$ (9:57–10:03), 95% within $$\mu \pm 2\sigma$$ (9:54–10:06) and 99.7% within $$\mu \pm 3\sigma$$ (9:51–10:09). Closing the door at $$\mu + 2\sigma$$ (10:06) shuts out about 2.3% of students, roughly 1 in 40; at $$\mu + 3\sigma$$ (10:09) only about 0.13%.

![Gaussian arrival times with μ = 10:00 and σ = 3 min, and the door closing at μ + 2σ](/modules/math_essentials/images/gaussian_arrivals.svg)

The nested bands are $$\mu \pm \sigma$$, $$2\sigma$$ and $$3\sigma$$; the red tail marks the students who still arrive after the door closes at $$\mu + 2\sigma$$.

## Test yourself

Go through the following tasks and questions. If you stumble upon huge obstacles, go back to the material above. Try each task before you open its solution.

1. Let $$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$, write down $$\mathbf{A}^T$$, calculate $$\mathbf{A}^{-1}$$ and check your result!

    <details class="notice--info" markdown="1">
    <summary><strong>Solution</strong></summary>

    $$\mathbf{A}^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$$. $$\det \mathbf{A} = 1 \cdot 4 - 2 \cdot 3 = -2$$, so $$\mathbf{A}^{-1} = -\frac{1}{2} \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix} = \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix}$$. Check: $$\mathbf{A}\mathbf{A}^{-1} = \begin{bmatrix} -2 + 3 & 1 - 1 \\ -6 + 6 & 3 - 2 \end{bmatrix} = \mathbf{I}$$.

    </details>

2. If $$\mathbf{x} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$, calculate $$\mathbf{A}\mathbf{x}$$, $$\mathbf{x}^T\mathbf{x}$$ and $$\mathbf{A}\mathbf{A}^T$$. Write down the shapes first.

    <details class="notice--info" markdown="1">
    <summary><strong>Solution</strong></summary>

    $$\mathbf{A}\mathbf{x}$$ is $$(2 \times 2)(2 \times 1) = 2 \times 1$$: $$(1 - 2, 3 - 4)^T = (-1, -1)^T$$. $$\mathbf{x}^T\mathbf{x}$$ is $$1 \times 1$$: $$1 + 1 = 2$$. $$\mathbf{A}\mathbf{A}^T$$ is $$2 \times 2$$: $$\begin{bmatrix} 5 & 11 \\ 11 & 25 \end{bmatrix}$$.

    </details>

3. A data matrix with 3 examples and 2 features is $$\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 0 & 1 \end{bmatrix}$$, and a linear model has weights $$\mathbf{w} = (2, -1)^T$$. Which of $$\mathbf{X}\mathbf{w}$$, $$\mathbf{w}\mathbf{X}$$, $$\mathbf{X}^T\mathbf{w}$$ and $$\mathbf{X}^T\mathbf{X}$$ are defined, and what are their shapes? Compute the predictions $$\mathbf{X}\mathbf{w}$$.

    <details class="notice--info" markdown="1">
    <summary><strong>Solution</strong></summary>

    $$\mathbf{X}\mathbf{w}$$ is $$(3 \times 2)(2 \times 1) = 3 \times 1$$. $$\mathbf{w}\mathbf{X}$$ is $$(2 \times 1)(3 \times 2)$$ and $$\mathbf{X}^T\mathbf{w}$$ is $$(2 \times 3)(2 \times 1)$$; neither is defined. $$\mathbf{X}^T\mathbf{X}$$ is $$(2 \times 3)(3 \times 2) = 2 \times 2$$. The predictions are $$\mathbf{X}\mathbf{w} = (2 - 2, 6 - 4, 0 - 1)^T = (0, 2, -1)^T$$, one per example.

    </details>

4. Let $$\mathbf{z} = (1, 1)^T$$. What are the angle, the cosine distance and the Euclidean distance between $$\mathbf{x}$$ and $$\mathbf{z}$$? How do the two distances change if you replace $$\mathbf{z}$$ by $$3\mathbf{z}$$?

    <details class="notice--info" markdown="1">
    <summary><strong>Solution</strong></summary>

    $$\mathbf{x}^T\mathbf{z} = 1 - 1 = 0$$, so $$\cos\theta = 0$$, $$\theta = 90°$$ and the cosine distance is 1. The Euclidean distance is $$\lVert(0, -2)\rVert_2 = 2$$. With $$3\mathbf{z}$$ the angle and the cosine distance stay the same, while the Euclidean distance grows to $$\lVert(-2, -4)\rVert_2 = \sqrt{20} \approx 4.47$$.

    </details>

5. Let $$y$$ be a random variable representing the height of a person and $$g$$ a discrete random variable representing the gender (assumed to be binary). What does the term $$p(y \mid g = \text{male})$$ refer to? What does $$p(g = \text{female})$$ refer to?

    <details class="notice--info" markdown="1">
    <summary><strong>Solution</strong></summary>

    $$p(y \mid g = \text{male})$$ is the conditional density of heights among males only. $$p(g = \text{female})$$ is the marginal probability that a randomly chosen person is female, regardless of height.

    </details>

6. Go back to the binary test, but now assume that 10% of people are sick (the test still detects 90% of sick people and is positive for 10% of healthy people). Fill the table for 1,000 people and compute $$p(K{=}1 \mid T{=}1)$$.

    <details class="notice--info" markdown="1">
    <summary><strong>Solution</strong></summary>

    100 people are sick, 90 of them test positive. 900 are healthy, 90 of them test positive. So 180 people test positive and $$p(K{=}1 \mid T{=}1) = 90 / 180 = 0.5$$. A positive result of the same test means much more when the disease is more common.

    </details>

7. Heights in a group are Gaussian with $$\mu = 170$$ cm and $$\sigma = 10$$ cm. About what fraction of the group is taller than 190 cm?

    <details class="notice--info" markdown="1">
    <summary><strong>Solution</strong></summary>

    190 cm is $$\mu + 2\sigma$$. About 95% lie within $$\mu \pm 2\sigma$$, and the remaining 5% split evenly between both tails, so about 2.5% (more precisely 2.3%) are taller than 190 cm.

    </details>

## Further reading

1. *Deisenroth, M. P., Faisal, A. A., & Ong, C. S.* (2020). "Mathematics for machine learning". Cambridge University Press - a whole book that compiles basic and advanced math concepts for machine learning

**Vectors and matrices**

- [📚MML Book, Sect. 2.1 & Sect. 2.2](https://mml-book.github.io/book/mml-book.pdf)
- [📚Khan Academy: matrices](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices)
- [📚Online test at Khan Academy: matrix multiplication](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/v/matrix-multiplication-intro)
- [⭐Intuitive view on linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

**Differentiation and integration**

- [⭐Geometric view on calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

**Distributions and probabilities**

- [📚Random variables](/modules/statistical-inference/stat-inf.md#probability-and-random-variables)
- [📚Probability distributions](/modules/statistical-inference/stat-inf.md#probability-distribution)
- [📺Learn about conditional distributions and Bayes' theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)
