#'train
#'
#'Trains a basic feedforward neural network using stochastic gradient descent.
#'Tracks loss history and returns network with updated weights and biases.
#'
#'@param network A list with attributes `layer_sizes`, `weights`, and `biases`.
#'@param X An n x k matrix-like object with n observations and k features
#'@param Y An n x 1 vector of outcome data.
#'@param epochs The number of training epochs during stochastic gradient descent.
#'@param mini_batch_size The size of the batches during stochastic gradient descent.
#'@param learning_rate The learning rate during stochastic gradient descent.
#'@return A trained network with updated weights and biases.
#'@examples
#'X=mtcars[, c("mpg","cyl", "hp")]
#'Y=mtcars[, "am"]
#'layer_sizes=c(3,9,4,1)
#'network=list(layer_sizes=layer_sizes,
#'             weights=initialize_weights(layer_sizes),
#'             biases=initialize_biases(layer_sizes))
#'train(network, X, Y, epochs=5000, mini_batch_size=6, learning_rate=0.01)
#'@export
train=function(network, X, Y, epochs=5000, mini_batch_size=32, learning_rate=0.01){
  n <- nrow(X)

  # Validate dimensions
  stopifnot(
    n == nrow(Y),
    ncol(X) == network$layer_sizes[1],
    ncol(Y) == tail(network$layer_sizes, 1),
    mini_batch_size >= 1,
    mini_batch_size <= n
  )

  # Retrieve parameters
  weights <- network$weights
  biases <- network$biases
  layer_sizes=network$layer_sizes

  trained=sgd(network, X, Y, epochs, mini_batch_size, learning_rate)


  trained_weights=trained$weights
  trained_biases=trained$biases
  loss_history=trained$loss_history

  print(loss_history[1])
  print(loss_history[epochs])

  return(list(layer_sizes=network$layer_sizes,
              weights=trained_weights,
              biases=trained_biases))
}





