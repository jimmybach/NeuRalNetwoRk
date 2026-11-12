run_backprop=function(layer_sizes, x, y, weights=NULL, biases=NULL){
  n_layers=length(layer_sizes)-1

  if(is.null(weights)){
    weights=initialize_weights(layer_sizes)
  }

  if(is.null(biases)){
    biases=initialize_biases(layer_sizes)
  }

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

  }

  delta=cost_derivative(activations[[n_layers+1]],y)*sigmoid_prime(activations[[n_layers+1]])
  bias_gradients[[n_layers]]=matrix(rowSums(delta),ncol=1)
  weight_gradients[[n_layers]]=delta %*% t(activations[[n_layers]])
  for(i in (n_layers-1):1){
    sp=sigmoid_prime(activations[[i+1]])
    delta=(t(weights[[i+1]]) %*% delta) * sp
    bias_gradients[[i]]=matrix(rowSums(delta),ncol=1)
    weight_gradients[[i]]=delta %*% t(activations[[i]])

  }

  return(list(weight_gradients,bias_gradients))

}


