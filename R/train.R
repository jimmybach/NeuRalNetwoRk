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

  trained=sgd(network, X, y, epochs, mini_batch_size, learning_rate)


  trained_weights=trained$weights
  trained_biases=trained$biases
  loss_history=trained$loss_history

  print(loss_history[1])
  print(loss_history[epochs])

  return(list(layer_sizes=network$layer_sizes,
              weights=trained_weights,
              biases=trained_biases))
}





