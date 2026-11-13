#'sigmoid
#'
#'Calculates the activation value of an input using the sigmoid function
#'
#'@param z input value
#'@return the sigmoid of z
#'@examples
#'sigmoid(3)
#'@export
sigmoid=function(z){
  return(1/(1+exp(-z)))
}

#'sigmoid_prime
#'
#'Calculates the derivative value of an input that is a sigmoid output.
#'
#'@param a An activation value from a sigmoid function.
#'@return The derivative of activation value.
#'@examples
#'sigmoid_prime((1/1+exp(-1)))
#'@export
sigmoid_prime = function(a){
  return(a * (1-a))
}
