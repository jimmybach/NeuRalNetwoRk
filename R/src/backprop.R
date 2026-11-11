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
    seq_len(length(layer_sizes) - 1),
    function(i) {
      matrix(rnorm(layer_sizes[i+1]),ncol=1)
    }
  )
}


for
sigmoid=function(z){
  return(1/(1+exp(-z)))
}

run_ffn=function(layer_sizes, x){
  weights=initialize_weights(layer_sizes)
  biases=initialize_biases(layer_sizes)
  activation=t(x)
  activations=numeric()
  zs=numeric()

  for(i in 1:(length(layer_sizes)-1)){
    z=weights[[i]]%*%activation + biases[[i]]%*%t(rep(1,ncol(activation)))
    append(zs, z)
    activation=sigmoid(z)
    append(activations, activation)
  }
}

