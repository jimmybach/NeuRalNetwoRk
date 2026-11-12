initialize_weights=function(layer_sizes){
  lapply(
    seq_len(length(layer_sizes) - 1),
    function(i) {
      x=layer_sizes[i]
      y=layer_sizes[i + 1]

      matrix(rnorm(y * x), nrow = y, ncol = x)
    }
  )
}

initialize_biases=function(layer_sizes){
  lapply(
    seq_len(length(layer_sizes)-1),
    function(i) {
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

initialize_network=function(layer_sizes, weights=NULL, biases=NULL){
  if(is.null(weights)){
    weights=initialize_weights(layer_sizes)
  }

  if(is.null(biases)){
    biases=initialize_biases(layer_sizes)
  }

  return(list(layer_sizes=layer_sizes,
              weights=weights,
              biases=biases))
}
