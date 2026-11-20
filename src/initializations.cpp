
#include <Rcpp.h>
#include <cmath>

using namespace Rcpp;

// Validate layer sizes
void validate_layers(IntegerVector layer_sizes) {
  if (layer_sizes.size() < 2) {
    stop("Network must contain at least two layers.");
  }

  for (int i = 0; i < layer_sizes.size(); i++) {
    if (layer_sizes[i] == NA_INTEGER || layer_sizes[i] < 1) {
      stop("All layer sizes must be positive integers.");
    }
  }
}


// Initialize weight matrices with N(0, 1) values
// [[Rcpp::export]]
List initialize_weights_cpp(IntegerVector layer_sizes) {

  validate_layers(layer_sizes);

  int n_layers = layer_sizes.size() - 1;
  List weights(n_layers);

  for (int i = 0; i < n_layers; i++) {

    int n_inputs = layer_sizes[i];
    int n_outputs = layer_sizes[i + 1];

    NumericMatrix W(n_outputs, n_inputs);

    // R matrices are stored column-major
    // Fill each element with a standard normal value
    NumericVector values = Rcpp::rnorm(n_outputs * n_inputs);

    std::copy(values.begin(), values.end(), W.begin());

    weights[i] = W;
  }

  return weights;
}


// Initialize bias vectors with N(0, 1) values
// [[Rcpp::export]]
List initialize_biases_cpp(IntegerVector layer_sizes) {

  validate_layers(layer_sizes);

  int n_layers = layer_sizes.size() - 1;
  List biases(n_layers);

  for (int i = 0; i < n_layers; i++) {

    int n_outputs = layer_sizes[i + 1];

    NumericMatrix b(n_outputs, 1);

    NumericVector values = Rcpp::rnorm(n_outputs);

    std::copy(values.begin(), values.end(), b.begin());

    biases[i] = b;
  }

  return biases;
}


// Initialize bias gradients to zero
// [[Rcpp::export]]
List initialize_bias_gradients_cpp(IntegerVector layer_sizes) {

  validate_layers(layer_sizes);

  int n_layers = layer_sizes.size() - 1;
  List gradients(n_layers);

  for (int i = 0; i < n_layers; i++) {

    int n_outputs = layer_sizes[i + 1];

    NumericMatrix db(n_outputs, 1);
    std::fill(db.begin(), db.end(), 0.0);

    gradients[i] = db;
  }

  return gradients;
}


// Initialize weight gradients to zero
// [[Rcpp::export]]
List initialize_weight_gradients_cpp(IntegerVector layer_sizes) {

  validate_layers(layer_sizes);

  int n_layers = layer_sizes.size() - 1;
  List gradients(n_layers);

  for (int i = 0; i < n_layers; i++) {

    int n_inputs = layer_sizes[i];
    int n_outputs = layer_sizes[i + 1];

    NumericMatrix dW(n_outputs, n_inputs);
    std::fill(dW.begin(), dW.end(), 0.0);

    gradients[i] = dW;
  }

  return gradients;
}


// Initialize a complete neural network
// [[Rcpp::export]]
List initialize_network_cpp(IntegerVector layer_sizes) {

  validate_layers(layer_sizes);

  List weights = initialize_weights_cpp(layer_sizes);
  List biases = initialize_biases_cpp(layer_sizes);

  return List::create(
    Named("layer_sizes") = layer_sizes,
    Named("weights") = weights,
    Named("biases") = biases
  );
}
