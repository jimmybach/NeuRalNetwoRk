test_that("training throws errors for improper parameters", {
  set.seed(123)
  X=matrix(mtcars[, c("mpg","cyl", "hp")])
  Y=matrix(mtcars[, "am"])

  #Test that a single layer throws an error
  layer_sizes=c(3)
  expect_error(initialize_network(layer_sizes))

  #Test that a layer with 0 neurons throws an error
  layer_sizes=c(3,0,55)
  expect_error(initialize_network(layer_sizes))

  #Test that a layer with a fractional amount of neurons throws an error
  layer_sizes=c(2.5,5,10)
  expect_error(initialize_network(layer_sizes))

  #Test that training a network with an input layer that has an amount of neurons
  #Different from the dimensionality of X throws an error
  layer_sizes=c(4,3,9,1)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y))

  #Test that training a network with an output layer that has an amount of neurons
  #Different from the dimensionality of Y throws an error
  layer_sizes=c(3,4,9,2)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y))

  #Test that training a network with 0 epochs throws an error
  layer_sizes=c(3,4,9,1)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y, epochs=0))

  #Test that training a network with a batch size of 0 throws an error
  layer_sizes=c(3,4,9,1)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y, epochs=50, mini_batch_size = 0))

  #Test that training a network with a batch size larger than the number
  #of individuals in the dataset throws an error (in this case, n=32)
  layer_sizes=c(3,4,9,1)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y, epochs=50, mini_batch_size = 500))

  #Test that training with a negative learning rate throws an error
  layer_sizes=c(3,4,9,1)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y, learning_rate = -.001))
})

