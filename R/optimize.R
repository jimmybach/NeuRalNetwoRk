sgd=function(network, X,y,epochs, mini_batch_size, learning_rate, verbose=TRUE){
  n=nrow(X)
  #initialize the loss history, which stores the prediction loss after each epoch
  loss_history=numeric(epochs)

  weights <- network$weights
  biases <- network$biases
  layer_sizes=network$layer_sizes


  for(epoch in 1:epochs){
    #split the data randomly into mini batches of size mini_batch_size
    indices=sample(n)
    mini_batches=split(indices, ceiling(seq_along(indices)/mini_batch_size))
    for(batch in mini_batches){

      batch_x=X[batch,]
      batch_y=y[batch]

      #Run gradient descent on a mini batch of data
      updated_params=update_mini_batch(layer_sizes, batch_x, batch_y, learning_rate, weights, biases)
      weights=updated_params[[1]]
      biases=updated_params[[2]]
    }

    #Calculate new predictions after the gradient descent on all mini batches
    predictions=feedforward(list(layer_sizes=layer_sizes,
                                 weights=weights,
                                 biases=biases), X)

    #Calculate the new loss using the new predictions
    loss=cost(t(predictions), y)

    #If desired, print out the loss of each epoch
    if(verbose){
      cat(sprintf("Epoch %s loss: %.4f\n", epoch, loss))
    }

    loss_history[epoch]=loss


  }
  return(list(weights=weights,
              biases=biases,
              loss_history=loss_history))
}

update_mini_batch=function(layer_sizes, batch_x, batch_y, learning_rate, weights, biases){

  #Update the gradients with backpropagation
  gradients=run_backprop(layer_sizes, batch_x, batch_y, weights, biases)
  weight_gradients=gradients[[1]]
  bias_gradients=gradients[[2]]

  n_layers=length(layer_sizes)-1
  for(i in 1:n_layers){

    #For each layer, update the weights and biases using gradient descent
    weights[[i]]=weights[[i]]-learning_rate*weight_gradients[[i]]
    biases[[i]]=biases[[i]]-learning_rate*bias_gradients[[i]]
  }

  return(list(weights, biases))
}
