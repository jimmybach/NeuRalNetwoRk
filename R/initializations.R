initialize_weights=function(layer_sizes){
  lapply(
    seq_len(length(layer_sizes) - 1),
    function(i) {
      x=layer_sizes[i]
      y=layer_sizes[i + 1]

      #The number of rows is equal to the layer size of the next dimension
      #The number of columns is equal to the layer size of the current dimension
      matrix(rnorm(y * x), nrow = y, ncol = x)
    }
  )
}


initialize_biases=function(layer_sizes){
  lapply(
    seq_len(length(layer_sizes)-1),
    function(i) {
      #The length of the bias vector is equal to the size of the next layer
      matrix(rnorm(layer_sizes[i+1]),ncol=1)
    }
  )
}


initialize_bias_gradients=function(layer_sizes){
  lapply(
    seq_len(length(layer_sizes)-1),
    function(i) {
      matrix(rep(0,layer_sizes[i+1]),ncol=1)
    }
  )
}

initialize_weight_gradients=function(layer_sizes){
  lapply(
    seq_len(length(layer_sizes) - 1),
    function(i) {
      x=layer_sizes[i]
      y=layer_sizes[i + 1]

      matrix(rep(0,(y * x)), nrow = y, ncol = x)
    }
  )
}

#'initialize_network
#'
#'Initializes a feedforward neural network using a Gaussian distribution for both weights and biases.
#'@param layer_sizes A vector of integers specifying the number of neurons within each layer.
#'@param weights Optional list of weight matrices. If NULL, weights
#'   are randomly initialized.
#'@param biases Optional list of bias vectors. If NULL, biases
#'   are randomly initialized.
#'@return A list of with the layer_sizes, a list of random weights,
#'   and a list of random biases for each neuron of each layer.
#'@examples
#'data(mtcars)
#'
#' X=scale(mtcars[, c("mpg", "cyl", "hp")])
#' Y=matrix(mtcars$am, ncol = 1)
#'
#' layer_sizes <- c(3, 9, 4, 1)
#'
#' set.seed(123)
#' network <- initialize_network(layer_sizes)
#'@export
initialize_network=function(layer_sizes){

  #Ensure multiple layers of positive integer number of neurons
  stopifnot(
    all(layer_sizes>=1),
    all(layer_sizes == floor(layer_sizes)),
    length(layer_sizes)>=2
  )

  #Initialize weights and biases to random numbers from a normal distribution
  weights=initialize_weights(layer_sizes)
  biases=initialize_biases(layer_sizes)

  return(list(layer_sizes=layer_sizes,
              weights=weights,
              biases=biases))
}
