run_backprop=function(layer_sizes, x, y, weights=NULL, biases=NULL){
  n_layers=length(layer_sizes)-1

  if(is.null(weights)){
    weights=initialize_weights(layer_sizes)
  }

  if(is.null(biases)){
    biases=initialize_biases(layer_sizes)
  }

  ##Track values for each activation and pre-activated values
  activation=t(x)
  activations=list(t(x))
  zs=list()


  bias_gradients=initialize_bias_gradients(layer_sizes)
  weight_gradients=initialize_weight_gradients(layer_sizes)

  for(i in 1:(n_layers)){

    #Z=Wx+B
    z=weights[[i]]%*%activation + biases[[i]]%*%t(rep(1,ncol(activation)))
    zs[[i]]=z

    #A=sigmoid(Wx+B)
    activation=sigmoid(z)
    activations[[i+1]]=activation

  }

  # Compute the derivative of the cost function with respect to the output activations
  # Multiply element-wise by the derivative of the sigmoid activation function
  delta = cost_derivative(activations[[n_layers+1]], y) *
    sigmoid_prime(activations[[n_layers+1]])

  # Sum the error terms across all observations in the mini-batch
  bias_gradients[[n_layers]] = matrix(rowSums(delta), ncol=1)

  # Multiply the output layer error terms by the transpose of the previous layer's activations
  weight_gradients[[n_layers]] = delta %*% t(activations[[n_layers]])

  # Iterate backward from the last hidden layer to the first hidden layer
  for(i in (n_layers-1):1){

    # Calculate the derivative of the sigmoid activation function
    # for each neuron in the current hidden layer
    sp = sigmoid_prime(activations[[i+1]])

    # Propagate the error backward from the next layer
    # Multiply the transpose of the next layer's weight matrix by its error term
    # Then multiply element-wise by the current layer's activation derivative
    delta = (t(weights[[i+1]]) %*% delta) * sp

    # Calculate the bias gradients for the current hidden layer
    # Sum the error terms across all observations in the mini-batch
    bias_gradients[[i]] = matrix(rowSums(delta), ncol=1)

    # Calculate the weight gradients for the current hidden layer
    # Multiply the current layer's error terms by the transpose
    # of the activations from the preceding layer
    weight_gradients[[i]] = delta %*% t(activations[[i]])

  }
  return(list(weight_gradients,bias_gradients))

}


