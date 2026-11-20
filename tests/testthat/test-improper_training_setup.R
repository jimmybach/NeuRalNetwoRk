test_that("training throws errors for improper parameters", {
  set.seed(123)
  X=matrix(mtcars[, c("mpg","cyl", "hp")])
  Y=matrix(mtcars[, "am"])

  layer_sizes=c(3)
  expect_error(initialize_network(layer_sizes))

  layer_sizes=c(3,0,55)
  expect_error(initialize_network(layer_sizes))

  layer_sizes=c(2.5,5,10)
  expect_error(initialize_network(layer_sizes))

  layer_sizes=c(4,3,9,1)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y))

  layer_sizes=c(3,4,9,2)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y))

  layer_sizes=c(3,4,9,1)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y, epochs=0))

  layer_sizes=c(3,4,9,1)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y, epochs=50, mini_batch_size = 0))

  layer_sizes=c(3,4,9,1)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y, epochs=50, mini_batch_size = 500))

  layer_sizes=c(3,4,9,1)
  network=initialize_network(layer_sizes)
  expect_error(train(network, X, Y, learning_rate = -.001))
})

