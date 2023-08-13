# RealNVP Implementation for 2D Density Estimation

This project presents an implementation of RealNVP (Real-valued Non-Volume Preserving), a powerful model within the family of normalizing flows. Utilizing PyTorch, this implementation enables the modeling of complex 2D distributions through the use of affine coupling layers.

## Table of Contents

- [Background](#background)
- [Project Description](#project-description)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Sources](#sources)

## Background

Normalizing flows, such as RealNVP, have revolutionized density estimation and generative modeling. By providing invertible transformations, they enable exact likelihood computation, efficient sampling, and accurate modeling of complex distributions. RealNVP's architecture makes it a significant and versatile tool in various applications, such as anomaly detection, data synthesis, and more.

## Project Description

The goal of this project is to implement RealNVP for modeling 2D distributions. It provides an in-depth exploration of the architectural components, including the affine coupling layers that define the model's behavior.

### Data Overview

The project utilizes synthetic 2D data generated through the `datasets.make_moons` function. The data is then normalized and transformed, ready to be modeled by the RealNVP implementation.

## Model Architecture

RealNVP's architecture is built upon a series of affine coupling layers that alternate between variables, performing invertible transformations. Below is a general overview of the key components:

### Affine Coupling Layer

The Affine Coupling Layer is a fundamental component of the RealNVP architecture. It performs an invertible transformation on a subset of the input variables, allowing the model to learn complex dependencies between variables.

#### Mathematical Formulation

Given an input \( x \), the affine coupling layer splits it into two parts, \( x_1 \) and \( x_2 \), according to a binary mask. The transformation is defined as follows:

\[
\begin{align*}
y_1 & = x_1 \\
y_2 & = x_2 \cdot \exp(s(x_1)) + t(x_1)
\end{align*}
\]

Here, \( s(x_1) \) and \( t(x_1) \) are scale and translation functions, respectively. They are implemented as neural networks with \( x_1 \) as the input.

The inverse transformation is computed as:

\[
\begin{align*}
x_1 & = y_1 \\
x_2 & = (y_2 - t(y_1)) \cdot \exp(-s(y_1))
\end{align*}
\]

The determinant of the Jacobian of this transformation is simply the exponential of the sum of the scaling factors, making likelihood computation tractable:

\[
\text{det}\, J = \exp \left( \sum s(x_1) \right)
\]

### RealNVP Model

RealNVP (Real-valued Non-Volume Preserving) is a type of normalizing flow that builds upon the concept of the affine coupling layer. It consists of a series of such layers, enabling the modeling of complex, multi-dimensional distributions.

#### Mathematical Formulation

Given an observed variable \( x \), RealNVP transforms it into a latent space variable \( z \) through a series of invertible transformations:

\[
z = f(x; \theta)
\]

The inverse transformation maps back from the latent space to the observed space:

\[
x = f^{-1}(z; \theta)
\]

Each transformation is parameterized by \( \theta \), which includes the weights and biases of the neural networks in the affine coupling layers.

The likelihood of the observed data is computed using the change of variables formula:

\[
p(x) = p(z) \cdot \left| \text{det} \frac{\partial f^{-1}}{\partial z} \right|
\]

Where \( p(z) \) is a simple distribution (e.g., standard normal) that serves as the prior for the latent space, and the determinant term accounts for the change of density due to the transformation.

By stacking multiple affine coupling layers and alternating the variables being transformed, RealNVP can capture intricate dependencies between variables and model highly non-linear distributions.

These mathematical properties make RealNVP a versatile and powerful tool for density estimation, generative modeling, and more. It retains exact invertibility while enabling complex transformations, bridging the gap between flexibility and tractability in modeling distributions.

## Results

![image](https://github.com/DimensionDweller/realnvp_implimentation/assets/75709283/12e9d978-b037-419c-b7b1-d16412851ce7)

![image](https://github.com/DimensionDweller/realnvp_implimentation/assets/75709283/b676f8a2-c247-4613-8452-3432f591222e)


![image](https://github.com/DimensionDweller/realnvp_implimentation/assets/75709283/d6088574-a101-48d2-9d1a-395e33d8b7ce)

![image](https://github.com/DimensionDweller/realnvp_implimentation/assets/75709283/80c9b7bf-b0f5-45af-bd28-f4b50ae31c94)




Visualizations of the transformations and model's behavior can be observed throughout the training process. The project provides insights into how RealNVP transforms the input data, offering an intuitive understanding of its functionality.

## Usage

Clone the repository to your local machine:

```bash
git clone https://github.com/YourUsername/RealNVP_2D.git
```

Navigate to the directory of the project:

```bash
cd RealNVP_2D
```

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

Run the main script to start training the model:

```bash
python train_realnvp.py
```

Adjust the hyperparameters as needed within the script.

## Future Work

The potential applications of RealNVP and normalizing flows are vast. Future work may include:

- Extending the implementation to higher-dimensional data.
- Exploring various architectures and coupling mechanisms.
- Integrating RealNVP into broader systems, such as variational inference frameworks.

## Sources

- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using Real NVP. Retrieved from [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)

