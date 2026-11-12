cost=function(output, y){
  n=length(y)
  return((1/(2*n))*sum((output-y)^2))
}

cost_derivative=function(output, y){
  n=length(y)
  return((output-y)/n)
}
