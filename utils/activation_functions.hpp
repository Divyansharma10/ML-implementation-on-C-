#ifndef ACTIVATION_FUNCTIONS_HPP
#define ACTIVATION_FUNCTIONS_HPP
#include <memory>
#include "matrix_operations.hpp"
#include <cmath>
#include <algorithm>
#include <functional>

namespace ml {

// Activation function types
enum class ActivationType {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    ELU,
    SOFTMAX,
    LINEAR,
    SWISH,
    GELU
};

// Base class for activation functions
template<typename T = double>
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;
    virtual Matrix<T> forward(const Matrix<T>& input) = 0;
    virtual Matrix<T> backward(const Matrix<T>& input) = 0;
    virtual ActivationType get_type() const = 0;
    virtual std::string get_name() const = 0;
};

// Sigmoid activation function
template<typename T = double>
class Sigmoid : public ActivationFunction<T> {
public:
    Matrix<T> forward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = T{1} / (T{1} + std::exp(-input(i, j)));
            }
        }
        return result;
    }
    
    Matrix<T> backward(const Matrix<T>& input) override {
        Matrix<T> sigmoid_out = forward(input);
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = sigmoid_out(i, j) * (T{1} - sigmoid_out(i, j));
            }
        }
        return result;
    }
    
    ActivationType get_type() const override { return ActivationType::SIGMOID; }
    std::string get_name() const override { return "Sigmoid"; }
};

// Tanh activation function
template<typename T = double>
class Tanh : public ActivationFunction<T> {
public:
    Matrix<T> forward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = std::tanh(input(i, j));
            }
        }
        return result;
    }
    
    Matrix<T> backward(const Matrix<T>& input) override {
        Matrix<T> tanh_out = forward(input);
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = T{1} - tanh_out(i, j) * tanh_out(i, j);
            }
        }
        return result;
    }
    
    ActivationType get_type() const override { return ActivationType::TANH; }
    std::string get_name() const override { return "Tanh"; }
};

// ReLU activation function
template<typename T = double>
class ReLU : public ActivationFunction<T> {
public:
    Matrix<T> forward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = std::max(T{0}, input(i, j));
            }
        }
        return result;
    }
    
    Matrix<T> backward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = input(i, j) > T{0} ? T{1} : T{0};
            }
        }
        return result;
    }
    
    ActivationType get_type() const override { return ActivationType::RELU; }
    std::string get_name() const override { return "ReLU"; }
};

// Leaky ReLU activation function
template<typename T = double>
class LeakyReLU : public ActivationFunction<T> {
private:
    T alpha_;
    
public:
    explicit LeakyReLU(T alpha = T{0.01}) : alpha_(alpha) {}
    
    Matrix<T> forward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = input(i, j) > T{0} ? input(i, j) : alpha_ * input(i, j);
            }
        }
        return result;
    }
    
    Matrix<T> backward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = input(i, j) > T{0} ? T{1} : alpha_;
            }
        }
        return result;
    }
    
    ActivationType get_type() const override { return ActivationType::LEAKY_RELU; }
    std::string get_name() const override { return "LeakyReLU"; }
};

// ELU (Exponential Linear Unit) activation function
template<typename T = double>
class ELU : public ActivationFunction<T> {
private:
    T alpha_;
    
public:
    explicit ELU(T alpha = T{1.0}) : alpha_(alpha) {}
    
    Matrix<T> forward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = input(i, j) > T{0} ? input(i, j) : 
                              alpha_ * (std::exp(input(i, j)) - T{1});
            }
        }
        return result;
    }
    
    Matrix<T> backward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = input(i, j) > T{0} ? T{1} : 
                              alpha_ * std::exp(input(i, j));
            }
        }
        return result;
    }
    
    ActivationType get_type() const override { return ActivationType::ELU; }
    std::string get_name() const override { return "ELU"; }
};

// Softmax activation function
template<typename T = double>
class Softmax : public ActivationFunction<T> {
public:
    Matrix<T> forward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        
        for (size_t i = 0; i < input.rows(); ++i) {
            // Find max for numerical stability
            T max_val = input(i, 0);
            for (size_t j = 1; j < input.cols(); ++j) {
                max_val = std::max(max_val, input(i, j));
            }
            
            // Compute exponentials
            T sum = T{0};
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = std::exp(input(i, j) - max_val);
                sum += result(i, j);
            }
            
            // Normalize
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) /= sum;
            }
        }
        return result;
    }
    
    Matrix<T> backward(const Matrix<T>& input) override {
        Matrix<T> softmax_out = forward(input);
        Matrix<T> result(input.rows(), input.cols());
        
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                result(i, j) = softmax_out(i, j) * (T{1} - softmax_out(i, j));
            }
        }
        return result;
    }
    
    ActivationType get_type() const override { return ActivationType::SOFTMAX; }
    std::string get_name() const override { return "Softmax"; }
};

// Linear activation function (identity)
template<typename T = double>
class Linear : public ActivationFunction<T> {
public:
    Matrix<T> forward(const Matrix<T>& input) override {
        return input;
    }
    
    Matrix<T> backward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        result.ones();
        return result;
    }
    
    ActivationType get_type() const override { return ActivationType::LINEAR; }
    std::string get_name() const override { return "Linear"; }
};

// Swish activation function
template<typename T = double>
class Swish : public ActivationFunction<T> {
public:
    Matrix<T> forward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                T sigmoid_val = T{1} / (T{1} + std::exp(-input(i, j)));
                result(i, j) = input(i, j) * sigmoid_val;
            }
        }
        return result;
    }
    
    Matrix<T> backward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                T sigmoid_val = T{1} / (T{1} + std::exp(-input(i, j)));
                result(i, j) = sigmoid_val * (T{1} + input(i, j) * (T{1} - sigmoid_val));
            }
        }
        return result;
    }
    
    ActivationType get_type() const override { return ActivationType::SWISH; }
    std::string get_name() const override { return "Swish"; }
};

// GELU activation function
template<typename T = double>
class GELU : public ActivationFunction<T> {
public:
    Matrix<T> forward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        const T sqrt_2_pi = std::sqrt(T{2} / M_PI);
        
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                T x = input(i, j);
                T tanh_arg = sqrt_2_pi * (x + T{0.044715} * x * x * x);
                result(i, j) = T{0.5} * x * (T{1} + std::tanh(tanh_arg));
            }
        }
        return result;
    }
    
    Matrix<T> backward(const Matrix<T>& input) override {
        Matrix<T> result(input.rows(), input.cols());
        const T sqrt_2_pi = std::sqrt(T{2} / M_PI);
        
        for (size_t i = 0; i < input.rows(); ++i) {
            for (size_t j = 0; j < input.cols(); ++j) {
                T x = input(i, j);
                T tanh_arg = sqrt_2_pi * (x + T{0.044715} * x * x * x);
                T tanh_val = std::tanh(tanh_arg);
                T sech2_val = T{1} - tanh_val * tanh_val;
                
                result(i, j) = T{0.5} * (T{1} + tanh_val) + 
                              T{0.5} * x * sech2_val * sqrt_2_pi * (T{1} + T{0.134145} * x * x);
            }
        }
        return result;
    }
    
    ActivationType get_type() const override { return ActivationType::GELU; }
    std::string get_name() const override { return "GELU"; }
};

// Factory function to create activation functions
template<typename T = double>
std::unique_ptr<ActivationFunction<T>> create_activation_function(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID:
            return std::make_unique<Sigmoid<T>>();
        case ActivationType::TANH:
            return std::make_unique<Tanh<T>>();
        case ActivationType::RELU:
            return std::make_unique<ReLU<T>>();
        case ActivationType::LEAKY_RELU:
            return std::make_unique<LeakyReLU<T>>();
        case ActivationType::ELU:
            return std::make_unique<ELU<T>>();
        case ActivationType::SOFTMAX:
            return std::make_unique<Softmax<T>>();
        case ActivationType::LINEAR:
            return std::make_unique<Linear<T>>();
        case ActivationType::SWISH:
            return std::make_unique<Swish<T>>();
        case ActivationType::GELU:
            return std::make_unique<GELU<T>>();
        default:
            throw std::invalid_argument("Unknown activation function type");
    }
}

// Utility functions for direct application without objects
template<typename T>
Matrix<T> apply_sigmoid(const Matrix<T>& input);

template<typename T>
Matrix<T> apply_relu(const Matrix<T>& input);

template<typename T>
Matrix<T> apply_tanh(const Matrix<T>& input);

template<typename T>
Matrix<T> apply_softmax(const Matrix<T>& input);

// Derivative functions
template<typename T>
Matrix<T> sigmoid_derivative(const Matrix<T>& input);

template<typename T>
Matrix<T> relu_derivative(const Matrix<T>& input);

template<typename T>
Matrix<T> tanh_derivative(const Matrix<T>& input);

} // namespace ml
// Add these functions to your activation_functions.hpp file
// or create a new activation_functions.cpp file with these implementations

namespace ml {

// Scalar activation functions
double activate(double x, ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID:
            return 1.0 / (1.0 + std::exp(-x));
        
        case ActivationType::TANH:
            return std::tanh(x);
        
        case ActivationType::RELU:
            return std::max(0.0, x);
        
        case ActivationType::LEAKY_RELU:
            return x > 0.0 ? x : 0.01 * x;
        
        case ActivationType::ELU:
            return x > 0.0 ? x : (std::exp(x) - 1.0);
        
        case ActivationType::LINEAR:
            return x;
        
        case ActivationType::SWISH:
            return x / (1.0 + std::exp(-x));
        
        case ActivationType::GELU: {
            const double sqrt_2_pi = std::sqrt(2.0 / M_PI);
            double tanh_arg = sqrt_2_pi * (x + 0.044715 * x * x * x);
            return 0.5 * x * (1.0 + std::tanh(tanh_arg));
        }
        
        case ActivationType::SOFTMAX:
            // Note: Softmax requires the full vector, not individual elements
            // This is a simplified version that just returns exp(x)
            return std::exp(x);
        
        default:
            throw std::invalid_argument("Unknown activation function type");
    }
}

// Scalar activation derivative functions
double activate_derivative(double x, ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: {
            double s = activate(x, ActivationType::SIGMOID);
            return s * (1.0 - s);
        }
        
        case ActivationType::TANH: {
            double t = std::tanh(x);
            return 1.0 - t * t;
        }
        
        case ActivationType::RELU:
            return x > 0.0 ? 1.0 : 0.0;
        
        case ActivationType::LEAKY_RELU:
            return x > 0.0 ? 1.0 : 0.01;
        
        case ActivationType::ELU:
            return x > 0.0 ? 1.0 : std::exp(x);
        
        case ActivationType::LINEAR:
            return 1.0;
        
        case ActivationType::SWISH: {
            double sigmoid_val = 1.0 / (1.0 + std::exp(-x));
            return sigmoid_val * (1.0 + x * (1.0 - sigmoid_val));
        }
        
        case ActivationType::GELU: {
            const double sqrt_2_pi = std::sqrt(2.0 / M_PI);
            double tanh_arg = sqrt_2_pi * (x + 0.044715 * x * x * x);
            double tanh_val = std::tanh(tanh_arg);
            double sech2_val = 1.0 - tanh_val * tanh_val;
            
            return 0.5 * (1.0 + tanh_val) + 
                   0.5 * x * sech2_val * sqrt_2_pi * (1.0 + 0.134145 * x * x);
        }
        
        case ActivationType::SOFTMAX:
            // Note: Softmax derivative is more complex and requires the full vector
            // This is a simplified version
            return std::exp(x);
        
        default:
            throw std::invalid_argument("Unknown activation function type");
    }
}

// Utility function to convert ActivationType to string
std::string activation_to_string(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: return "Sigmoid";
        case ActivationType::TANH: return "Tanh";
        case ActivationType::RELU: return "ReLU";
        case ActivationType::LEAKY_RELU: return "LeakyReLU";
        case ActivationType::ELU: return "ELU";
        case ActivationType::SOFTMAX: return "Softmax";
        case ActivationType::LINEAR: return "Linear";
        case ActivationType::SWISH: return "Swish";
        case ActivationType::GELU: return "GELU";
        default: return "Unknown";
    }
}

} // namespace ml
#endif // ACTIVATION_FUNCTIONS_HPP
