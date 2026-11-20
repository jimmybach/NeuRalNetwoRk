test_that("initialization works", {
  layer_sizes=c(4, 3, 9, 1)

  network=initialize_network(layer_sizes)
  expect_equal(network$layer_sizes, layer_sizes)

  weight_grads=initialize_weight_gradients(layer_sizes)
  bias_grads=initialize_bias_gradients(layer_sizes)

  for(i in 1:(length(layer_sizes)-1)){
    expect_equal(dim(network$weights[[i]]), c(layer_sizes[i+1],layer_sizes[i]))
    expect_equal(dim(network$biases[[i]]), c(layer_sizes[i+1], 1))

    expect_equal(dim(weight_grads[[i]]), c(layer_sizes[i+1],layer_sizes[i]))
    expect_equal(dim(bias_grads[[i]]), c(layer_sizes[i+1], 1))

    expect_true(all(weight_grads[[i]]==0))
    expect_true(all(bias_grads[[i]]==0))


  }
})

