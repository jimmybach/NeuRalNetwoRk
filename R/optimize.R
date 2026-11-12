sgd=function(network, X,y,epochs, mini_batch_size, learning_rate){
  n=nrow(X)
  loss_history=numeric(epochs)

  weights <- network$weights
  biases <- network$biases
  layer_sizes=network$layer_sizes

  if(is.null(weights)){
    weights=initialize_weights(layer_sizes)
  }

  if(is.null(biases)){
    biases=initialize_biases(layer_sizes)
  }

  for(epoch in 1:epochs){
    indices=sample(n)
    mini_batches=split(indices, ceiling(seq_along(indices)/mini_batch_size))
    for(batch in mini_batches){

      batch_x=X[batch,]
      batch_y=y[batch]

      updated_params=update_mini_batch(layer_sizes, batch_x, batch_y, learning_rate, weights, biases)
      weights=updated_params[[1]]
      biases=updated_params[[2]]
    }

    predictions=feedforward(X, weights, biases)

    loss_history[epoch]=cost(predictions, y)
  }
  return(list(weights=weights,
              biases=biases,
              loss_history=loss_history))
}

update_mini_batch=function(layer_sizes, batch_x, batch_y, learning_rate, weights, biases){
  bias_gradients=initialize_bias_gradients(layer_sizes)
  weight_gradients=initialize_weight_gradients(layer_sizes)


  gradients=run_backprop(layer_sizes, batch_x, batch_y, weights, biases)
  delta_w_grad=gradients[[1]]
  delta_b_grad=gradients[[2]]

  n_layers=length(layer_sizes)-1
  for(i in 1:n_layers){

    weight_gradients[[i]]=weight_gradients[[i]]+delta_w_grad[[i]]
    bias_gradients[[i]]=bias_gradients[[i]]+delta_b_grad[[i]]

    weights[[i]]=weights[[i]]-learning_rate*weight_gradients[[i]]
    biases[[i]]=biases[[i]]-learning_rate*bias_gradients[[i]]
  }

  return(list(weights, biases))
}
