feedforward=function(X, weights, biases){
  a=t(X)
  for(i in 1:length(weights)){
    a=sigmoid(weights[[i]]%*%a + biases[[i]]%*%t(rep(1,ncol(a))))
  }
  return(a)
}

