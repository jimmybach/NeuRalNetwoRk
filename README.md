<!-- badges: start -->
  [![R-CMD-check](https://github.com/jimmybach/NeuRalNetwoRk/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/jimmybach/NeuRalNetwoRk/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

# NeuRalNetwoRk

**A feedforward neural network implementation built from scratch in R.**

## Overview

`NeuRalNetwoRk` is an R package for building and training fully connected feedforward neural networks using base R matrix operations.

The package implements the fundamental components of neural network training, including forward propagation, backpropagation, gradient calculation, and mini-batch stochastic gradient descent.

The project emphasizes understanding the underlying mathematics and optimization algorithms.

## Features

- **Custom network architectures:** Specify layer sizes and multiple hidden layers.
- **Forward propagation:** Generate predictions using vectorized matrix operations.
- **Backpropagation:** Calculate gradients with respect to weights and biases.
- **Mini-batch gradient descent:** Train models using configurable learning rates, batch sizes, and epochs.
- **Sigmoid activation:** Support sigmoid activations and their derivatives.
- **Quadratic cost:** Optimize a squared-error objective.
- **Vectorized computation:** Process multiple observations simultaneously.

## Installation

Load the package during development:

```r
devtools::load_all()
```

Or install it locally:

```r
devtools::install()
```

## Neural Network Architecture

Networks are defined using a vector of layer sizes:

```r
layer_sizes <- c(3, 9, 4, 1)
```

This creates a network with 3 input neurons, hidden layers of 9 and 4 neurons, and 1 output neuron.

For each trainable layer $l$, the weight matrix and bias vector have dimensions:

$$
W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}
$$

$$
b^{(l)} \in \mathbb{R}^{n_l \times 1}
$$

### Forward Propagation

Each layer computes:

$$
Z^{(l)} = W^{(l)}A^{(l-1)} + b^{(l)}\mathbf{1}^{T}
$$

$$
A^{(l)} = \sigma(Z^{(l)})
$$

The sigmoid activation function is:

$$
\sigma(z) = \frac{1}{1+e^{-z}}
$$

Observations are internally represented as columns of activation matrices.

### Backpropagation

Backpropagation calculates the gradients of the cost function using the chain rule.

For hidden layer $l$, the error is propagated using:

$$
\delta^{(l)} =
\left((W^{(l+1)})^T\delta^{(l+1)}\right)
\odot \sigma'(Z^{(l)})
$$

For a mini-batch, let $\Delta^{(l)}$ represent the matrix of error signals for layer $l$, including the appropriate loss scaling.

The weight gradient is:

$$
\nabla_{W^{(l)}}C =
\Delta^{(l)}(A^{(l-1)})^T
$$

The bias gradient is:

$$
\nabla_{b^{(l)}}C =
\Delta^{(l)}\mathbf{1}
$$

Here, $\odot$ denotes elementwise multiplication, and $\mathbf{1}$ is a vector of ones with length equal to the mini-batch size.

## Cost Function

The package uses a quadratic cost function.

For $m$ observations and $p$ output neurons:

$$
C = \frac{1}{2pm}
\sum_{i=1}^{m}\sum_{j=1}^{p}
(\hat{Y}_{ji}-Y_{ji})^2
$$

The derivative with respect to the predictions is:

$$
\nabla_{\hat{Y}}C =
\frac{1}{pm}(\hat{Y}-Y)
$$

This scaling is incorporated during backpropagation, so gradients do not need to be averaged again during parameter updates.

## Training

Training uses mini-batch gradient descent.

For each layer $l$, parameters are updated according to:

$$
W^{(l)} \leftarrow W^{(l)}-\eta\nabla_{W^{(l)}}C
$$

$$
b^{(l)} \leftarrow b^{(l)}-\eta\nabla_{b^{(l)}}C
$$

Here, $\eta$ represents the learning rate.

During each epoch, the training procedure:

1. Shuffles the training observations.
2. Divides the dataset into mini-batches.
3. Computes predictions and gradients for each mini-batch.
4. Updates the weights and biases sequentially.
5. Repeats until the specified number of epochs is complete.

The batch size can be configured to perform stochastic, mini-batch, or full-batch gradient descent.

## Example: Binary Classification

The following example demonstrates the intended high-level API using the built-in `mtcars` dataset. Function names should be adjusted to match the implemented package API.

```r
library(NeuRalNetwoRk)

data(mtcars)

# Prepare input features and target
X <- scale(mtcars[, c("wt", "hp", "disp")])
Y <- matrix(mtcars$am, ncol = 1)

# Initialize the network
network <- initialize_network(c(3, 9, 4, 1))

# Train the network
network <- train_network(
  network,
  X,
  Y,
  epochs = 500,
  batch_size = 8,
  learning_rate = 0.01
)

# Generate predictions
predictions <- forward(
  network$weights,
  network$biases,
  X
)

# Convert probabilities to classes
predicted_classes <- as.integer(predictions >= 0.5)
```

The target variable `am` represents transmission type:

- `0`: Automatic
- `1`: Manual

The sigmoid output produces values between 0 and 1, which can be interpreted as predicted probabilities of manual transmission.

The small `mtcars` dataset is primarily useful for testing the implementation.

## Package Organization

```text
NeuRalNetwoRk/
├── R/
│   ├── initialization.R
│   ├── activations.R
│   ├── losses.R
│   ├── forward.R
│   ├── backprop.R
│   ├── optimizers.R
│   └── train.R
├── tests/
│   └── testthat/
├── DESCRIPTION
├── NAMESPACE
└── README
