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

initialize_bias_gradients=function(layer_sizes){
  lapply(
    seq_len(length(layer_sizes) - 1),
    function(i) {
      matrix(rep(0,layer_sizes[i]),ncol=1)
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

sigmoid=function(z){
  return(1/(1+exp(-z)))
}

sigmoid_prime = function(a){
  return(a * (1-a))
}

cost=function(output, y){
  n=length(y)
  return((1/(2*n))*sum((output-y)^2))
}

cost_derivative=function(output, y){
  n=length(y)
  return((output-y)/n)
}
t(activations[[n_layers+1]])-Y
Y
run_backprop=function(layer_sizes, x, y){
  n_layers=length(layer_sizes)-1

  weights=initialize_weights(layer_sizes)
  biases=initialize_biases(layer_sizes)
  activation=t(x)
  activations=list(t(x))
  zs=list()

  bias_gradients=initialize_bias_gradients(layer_sizes)
  weight_gradients=initialize_weight_gradients(layer_sizes)

  for(i in 1:(n_layers)){


    z=weights[[i]]%*%activation + biases[[i]]%*%t(rep(1,ncol(activation)))
    zs[[i]]=z

    activation=sigmoid(z)
    activations[[i+1]]=activation
    print(activation)

  }


  delta=cost_derivative(activations[[n_layers+1]],t(y))*sigmoid_prime(zs[[n_layers]])
  bias_gradients[[n_layers]]=delta
  weight_gradients[[n_layers]]=delta %*% t(activations[[n_layers]])

  for(i in (n_layers-1):1){
    z=zs[[i]]
    sp=sigmoid_prime(z)
    delta=(t(weights[[i+1]]) %*% delta) * sp
    bias_gradients[[i]]=delta
    print(dim(delta))
    print(dim(activations[[i]]))
    weight_gradients[[i]]=delta %*% t(activations[[i]])

  }


  return(list(weight_gradients,bias_gradients))

}
