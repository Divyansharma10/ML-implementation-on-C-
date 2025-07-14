// #ifndef MLP_HPP
// #define MLP_HPP

// #include <vector>
// #include <string>
// #include <memory>
// #include "../utils/matrix_operations.hpp"
// #include "../utils/activation_functions.hpp"

// /**
//  * Multi-Layer Perceptron (MLP) Neural Network
//  * 
//  * This class implements a feedforward neural network with configurable
//  * layers, activation functions, and training capabilities.
//  */
// class MLP {
// private:
//     // Network architecture
//     std::vector<int> layer_sizes;           // Number of neurons in each layer
//     std::vector<std::vector<std::vector<double>>> weights;  // Weight matrices
//     std::vector<std::vector<double>> biases;                // Bias vectors
    
//     // Activation functions
//     std::vector<ml::ActivationType> activations;    // Activation function for each layer
    
//     // Training parameters
//     double learning_rate;
//     int epochs;
//     double loss_threshold;
    
//     // Training history
//     std::vector<double> loss_history;
    
//     // Forward propagation storage
//     std::vector<std::vector<double>> layer_outputs;     // Output of each layer
//     std::vector<std::vector<double>> layer_activations; // Activated outputs
    
// public:
//     /**
//      * Constructor
//      * @param layer_sizes Vector containing the number of neurons in each layer
//      * @param activations Vector of activation functions for each hidden/output layer
//      * @param learning_rate Learning rate for gradient descent
//      */
//     MLP(const std::vector<int>& layer_sizes, 
//         const std::vector<ml::ActivationType>& activations,
//         double learning_rate = 0.01);
    
//     /**
//      * Destructor
//      */
//     ~MLP();
    
//     // Network initialization
//     /**
//      * Initialize weights and biases randomly
//      * @param seed Random seed for reproducibility
//      */
//     void initialize_weights(int seed = 42);
    
//     /**
//      * Initialize weights using Xavier/Glorot initialization
//      */
//     void xavier_initialization();
    
//     /**
//      * Initialize weights using He initialization
//      */
//     void he_initialization();
    
//     // Forward propagation
//     /**
//      * Forward pass through the network
//      * @param input Input vector
//      * @return Output vector
//      */
//     std::vector<double> forward(const std::vector<double>& input);
    
//     /**
//      * Predict output for given input
//      * @param input Input vector
//      * @return Predicted output
//      */
//     std::vector<double> predict(const std::vector<double>& input);
    
//     // Backward propagation
//     /**
//      * Backward pass - compute gradients
//      * @param input Input vector
//      * @param target Target output vector
//      * @return Loss value
//      */
//     double backward(const std::vector<double>& input, 
//                    const std::vector<double>& target);
    
//     // Training methods
//     /**
//      * Train the network on a single example
//      * @param input Input vector
//      * @param target Target output vector
//      * @return Loss value
//      */
//     double train_single(const std::vector<double>& input, 
//                        const std::vector<double>& target);
    
//     /**
//      * Train the network on a batch of examples
//      * @param inputs Vector of input vectors
//      * @param targets Vector of target output vectors
//      * @param epochs Number of training epochs
//      * @return Final loss value
//      */
//     double train_batch(const std::vector<std::vector<double>>& inputs,
//                       const std::vector<std::vector<double>>& targets,
//                       int epochs = 1000);
    
//     /**
//      * Train with early stopping
//      * @param inputs Training input data
//      * @param targets Training target data
//      * @param val_inputs Validation input data
//      * @param val_targets Validation target data
//      * @param max_epochs Maximum number of epochs
//      * @param patience Number of epochs to wait for improvement
//      * @return Final loss value
//      */
//     double train_with_validation(const std::vector<std::vector<double>>& inputs,
//                                 const std::vector<std::vector<double>>& targets,
//                                 const std::vector<std::vector<double>>& val_inputs,
//                                 const std::vector<std::vector<double>>& val_targets,
//                                 int max_epochs = 1000,
//                                 int patience = 50);
    
//     // Loss functions
//     /**
//      * Calculate Mean Squared Error
//      * @param predicted Predicted output
//      * @param target Target output
//      * @return MSE loss
//      */
//     double mse_loss(const std::vector<double>& predicted, 
//                    const std::vector<double>& target);
    
//     /**
//      * Calculate Cross-Entropy loss
//      * @param predicted Predicted output
//      * @param target Target output
//      * @return Cross-entropy loss
//      */
//     double cross_entropy_loss(const std::vector<double>& predicted, 
//                              const std::vector<double>& target);
    
//     // Evaluation methods
//     /**
//      * Calculate accuracy for classification problems
//      * @param inputs Test input data
//      * @param targets Test target data
//      * @return Accuracy percentage
//      */
//     double calculate_accuracy(const std::vector<std::vector<double>>& inputs,
//                              const std::vector<std::vector<double>>& targets);
    
//     /**
//      * Evaluate the network on test data
//      * @param inputs Test input data
//      * @param targets Test target data
//      * @return Average loss
//      */
//     double evaluate(const std::vector<std::vector<double>>& inputs,
//                    const std::vector<std::vector<double>>& targets);
    
//     // Utility methods
//     /**
//      * Get network architecture information
//      * @return String description of the network
//      */
//     std::string get_architecture_info() const;
    
//     /**
//      * Get training history
//      * @return Vector of loss values during training
//      */
//     const std::vector<double>& get_loss_history() const;
    
//     /**
//      * Set learning rate
//      * @param lr New learning rate
//      */
//     void set_learning_rate(double lr);
    
//     /**
//      * Get current learning rate
//      * @return Current learning rate
//      */
//     double get_learning_rate() const;
    
//     /**
//      * Save model to file
//      * @param filename File to save the model
//      * @return true if successful, false otherwise
//      */
//     bool save_model(const std::string& filename) const;
    
//     /**
//      * Load model from file
//      * @param filename File to load the model from
//      * @return true if successful, false otherwise
//      */
//     bool load_model(const std::string& filename);
    
//     /**
//      * Print network summary
//      */
//     void print_summary() const;
    
//     /**
//      * Reset the network (reinitialize weights)
//      */
//     void reset();
    
// private:
//     /**
//      * Helper function to compute gradients for weights and biases
//      * @param layer_idx Layer index
//      * @param delta Error delta for this layer
//      * @param prev_activation Previous layer activation
//      * @return Weight gradients
//      */
//     std::vector<std::vector<double>> compute_weight_gradients(
//         int layer_idx, 
//         const std::vector<double>& delta, 
//         const std::vector<double>& prev_activation);
    
//     /**
//      * Helper function to compute error delta for previous layer
//      * @param layer_idx Current layer index
//      * @param current_delta Current layer delta
//      * @return Previous layer delta
//      */
//     std::vector<double> compute_prev_delta(
//         int layer_idx, 
//         const std::vector<double>& current_delta);
    
//     /**
//      * Apply weight updates
//      * @param layer_idx Layer index
//      * @param weight_gradients Weight gradients
//      * @param bias_gradients Bias gradients
//      */
//     void update_weights(int layer_idx,
//                        const std::vector<std::vector<double>>& weight_gradients,
//                        const std::vector<double>& bias_gradients);
    
//     /**
//      * Validate network architecture
//      * @return true if valid, false otherwise
//      */
//     bool validate_architecture() const;
// };

// #endif // MLP_HPP






#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <iostream>

class LinearRegression {
private:
    double weight;  // slope (m)
    double bias;    // intercept (b)
    double learning_rate;
    
public:
    // Constructor
    LinearRegression(double lr = 0.01) : weight(0.0), bias(0.0), learning_rate(lr) {}
    
    // Train the model using gradient descent
    void train(const std::vector<double>& X, const std::vector<double>& y, int epochs = 1000) {
        int n = X.size();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass - calculate predictions
            std::vector<double> predictions = predict(X);
            
            // Calculate gradients
            double weight_gradient = 0.0;
            double bias_gradient = 0.0;
            
            for (int i = 0; i < n; i++) {
                double error = predictions[i] - y[i];
                weight_gradient += error * X[i];
                bias_gradient += error;
            }
            
            weight_gradient /= n;
            bias_gradient /= n;
            
            // Update parameters
            weight -= learning_rate * weight_gradient;
            bias -= learning_rate * bias_gradient;
            
            // Print progress every 100 epochs
            if (epoch % 100 == 0) {
                double cost = calculateCost(X, y);
                std::cout << "Epoch " << epoch << ", Cost: " << cost << std::endl;
            }
        }
    }
    
    // Make predictions
    std::vector<double> predict(const std::vector<double>& X) const {
        std::vector<double> predictions;
        for (double x : X) {
            predictions.push_back(weight * x + bias);
        }
        return predictions;
    }
    
    // Predict single value
    double predict(double x) const {
        return weight * x + bias;
    }
    
    // Calculate cost (Mean Squared Error)
    double calculateCost(const std::vector<double>& X, const std::vector<double>& y) const {
        std::vector<double> predictions = predict(X);
        double cost = 0.0;
        int n = X.size();
        
        for (int i = 0; i < n; i++) {
            double error = predictions[i] - y[i];
            cost += error * error;
        }
        
        return cost / (2 * n);
    }
    
    // Get trained parameters
    double getWeight() const { return weight; }
    double getBias() const { return bias; }
    
    // Print model equation
    void printModel() const {
        std::cout << "Trained model: y = " << weight << "x + " << bias << std::endl;
    }
};

#endif // LINEAR_REGRESSION_H