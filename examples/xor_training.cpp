#include "../algorithms/mlp.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "=== XOR Problem Training ===" << std::endl;
    
    // XOR dataset
    std::vector<std::vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    
    std::vector<std::vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };
    
    // Create MLP: 2 inputs -> 4 hidden (ReLU) -> 1 output (Sigmoid)
    std::vector<int> layer_sizes = {2, 4, 1};
    std::vector<ml::ActivationType> activations = {ml::ActivationType::RELU, ml::ActivationType::SIGMOID};
    
    MLP network(layer_sizes, activations, 0.1);
    
    // Print network summary
    network.print_summary();
    std::cout << std::endl;
    
    // Test initial predictions
    std::cout << "=== Initial Predictions ===" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> prediction = network.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] -> "
                  << "Prediction: " << std::fixed << std::setprecision(4) << prediction[0] 
                  << " (Expected: " << targets[i][0] << ")" << std::endl;
    }
    std::cout << std::endl;
    
    // Train the network
    std::cout << "=== Training ===" << std::endl;
    double final_loss = network.train_batch(inputs, targets, 1000);
    std::cout << "Final loss: " << std::fixed << std::setprecision(6) << final_loss << std::endl;
    std::cout << std::endl;
    
    // Test final predictions
    std::cout << "=== Final Predictions ===" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> prediction = network.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] -> "
                  << "Prediction: " << std::fixed << std::setprecision(4) << prediction[0] 
                  << " (Expected: " << targets[i][0] << ")" << std::endl;
    }
    
    // Calculate accuracy
    double accuracy = network.calculate_accuracy(inputs, targets);
    std::cout << "\nAccuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    
    // Test model save/load
    std::cout << "\n=== Testing Model Save/Load ===" << std::endl;
    if (network.save_model("../models/xor_model.txt")) {
        std::cout << "Model saved successfully!" << std::endl;
        
        // Create new network and load model
        MLP loaded_network({2, 4, 1}, {ml::ActivationType::RELU, ml::ActivationType::SIGMOID}, 0.1);
        if (loaded_network.load_model("../models/xor_model.txt")) {
            std::cout << "Model loaded successfully!" << std::endl;
            
            // Test loaded model
            std::cout << "Testing loaded model:" << std::endl;
            for (size_t i = 0; i < inputs.size(); i++) {
                std::vector<double> prediction = loaded_network.predict(inputs[i]);
                std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] -> "
                          << "Prediction: " << std::fixed << std::setprecision(4) << prediction[0] << std::endl;
            }
        } else {
            std::cout << "Failed to load model!" << std::endl;
        }
    } else {
        std::cout << "Failed to save model!" << std::endl;
    }
    
    return 0;
}