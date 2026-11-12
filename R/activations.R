sigmoid=function(z){
  return(1/(1+exp(-z)))
}

sigmoid_prime = function(a){
  return(a * (1-a))
}
