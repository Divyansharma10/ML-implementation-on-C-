#include "activation_functions.hpp"

namespace ml {

// Utility functions for direct application without objects
template<typename T>
Matrix<T> apply_sigmoid(const Matrix<T>& input) {
    Sigmoid<T> sigmoid;
    return sigmoid.forward(input);
}

template<typename T>
Matrix<T> apply_relu(const Matrix<T>& input) {
    ReLU<T> relu;
    return relu.forward(input);
}

template<typename T>
Matrix<T> apply_tanh(const Matrix<T>& input) {
    Tanh<T> tanh;
    return tanh.forward(input);
}

template<typename T>
Matrix<T> apply_softmax(const Matrix<T>& input) {
    Softmax<T> softmax;
    return softmax.forward(input);
}

// Derivative functions
template<typename T>
Matrix<T> sigmoid_derivative(const Matrix<T>& input) {
    Sigmoid<T> sigmoid;
    return sigmoid.backward(input);
}

template<typename T>
Matrix<T> relu_derivative(const Matrix<T>& input) {
    ReLU<T> relu;
    return relu.backward(input);
}

template<typename T>
Matrix<T> tanh_derivative(const Matrix<T>& input) {
    Tanh<T> tanh;
    return tanh.backward(input);
}

// Explicit template instantiations for common types
template Matrix<float> apply_sigmoid<float>(const Matrix<float>&);
template Matrix<double> apply_sigmoid<double>(const Matrix<double>&);

template Matrix<float> apply_relu<float>(const Matrix<float>&);
template Matrix<double> apply_relu<double>(const Matrix<double>&);

template Matrix<float> apply_tanh<float>(const Matrix<float>&);
template Matrix<double> apply_tanh<double>(const Matrix<double>&);

template Matrix<float> apply_softmax<float>(const Matrix<float>&);
template Matrix<double> apply_softmax<double>(const Matrix<double>&);

template Matrix<float> sigmoid_derivative<float>(const Matrix<float>&);
template Matrix<double> sigmoid_derivative<double>(const Matrix<double>&);

template Matrix<float> relu_derivative<float>(const Matrix<float>&);
template Matrix<double> relu_derivative<double>(const Matrix<double>&);

template Matrix<float> tanh_derivative<float>(const Matrix<float>&);
template Matrix<double> tanh_derivative<double>(const Matrix<double>&);

} // namespace ml