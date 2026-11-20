cost=function(output, y){
  n=length(y)

  #Quadratic cost function
  return((1/(2*n))*sum((output-y)^2))
}

cost_derivative=function(output, y){
  n=length(y)

  #Quadratic cost derivative
  return((output-y)/n)
}
