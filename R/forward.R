#'feedforward
#'
#'Makes a forward pass to a network of weights and biases using data.
#'@param network A list with attributes `layer_sizes`, `weights`, and `biases`.
#'@param X An n x k matrix-like object with n observations and k features.
#'@return Outputs of the forward pass of `X` to `network`.
#'@examples
#'X=mtcars[, c("mpg","cyl", "hp")]
#'Y=mtcars[, "am"]
#'layer_sizes=c(3,9,4,1)
#'network=initialize_network(layer_sizes)
#'feedforward(network, X)
#'@export
feedforward=function(network, X){
  a=t(X)
  weights=network$weights
  biases=network$biases

  for(i in 1:length(weights)){
    a=sigmoid(weights[[i]]%*%a + biases[[i]]%*%t(rep(1,ncol(a))))
  }
  return(a)
}
