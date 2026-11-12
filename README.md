# YourPackageName

**A feedforward neural network implementation built from scratch in R.**

## Overview

`YourPackageName` is an R package for building and training fully connected feedforward neural networks using base R matrix operations.

The package implements the fundamental components of neural network training, including forward propagation, backpropagation, gradient calculation, and mini-batch stochastic gradient descent.

The project is designed to provide a transparent, educational implementation of neural networks, with an emphasis on understanding the underlying mathematics and optimization algorithms.

## Features

- **Custom network architectures:** Specify the number of neurons in each layer, including multiple hidden layers.
- **Feedforward propagation:** Compute neural network predictions using matrix multiplication and activation functions.
- **Backpropagation:** Calculate analytical gradients of the cost function with respect to weights and biases.
- **Mini-batch gradient descent:** Train models using configurable learning rates, batch sizes, and epochs.
- **Sigmoid activation:** Apply sigmoid activation and its derivative during forward and backward propagation.
- **Quadratic cost function:** Optimize a squared-error objective using gradient descent.
- **Vectorized computations:** Process multiple training observations simultaneously using R matrix operations.

## Installation

During development, the package can be loaded from its project directory:

```r
devtools::load_all()
```

Alternatively, install the package locally:

```r
devtools::install()
```

## Neural Network Architecture

Neural network architectures are specified using a vector of layer sizes.

For example:

```r
layer_sizes <- c(3, 9, 4, 1)
```

This creates a fully connected network with:

- 3 input neurons
- 9 neurons in the first hidden layer
- 4 neurons in the second hidden layer
- 1 output neuron

For each trainable layer \(l\), the package initializes a weight matrix and bias vector:

\[
W^{(l)}\in\mathbb R^{n_l\times n_{l-1}}
\]

\[
b^{(l)}\in\mathbb R^{n_l\times1}
\]

### Forward Propagation

Each layer computes:

\[
Z^{(l)}=W^{(l)}A
