test_that("results are reproducible with random seed", {
  set.seed(123)
  X=mtcars[, c("mpg","cyl", "hp")]
  Y=mtcars[, "am"]
  layer_sizes=c(3,9,4,1)
  network=initialize_network(layer_sizes)
  trained_network=train(network, X, Y, epochs=500, mini_batch_size=6, learning_rate=0.01, verbose=FALSE)
  predictions=feedforward(trained_network, X)

  expect_equal(cost(t(predictions), Y),0.1190849, tolerance=1e-6)

})
