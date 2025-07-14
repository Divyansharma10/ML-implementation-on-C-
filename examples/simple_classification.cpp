// #include "../algorithms/mlp.hpp"
// #include <iostream>
// #include <vector>
// #include <random>
// #include <cmath>
// #include <iomanip>

// // Generate synthetic 2D classification data
// void generate_spiral_data(std::vector<std::vector<double>>& inputs, 
//                          std::vector<std::vector<double>>& targets, 
//                          int points_per_class = 100) {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::normal_distribution<> noise(0.0, 0.1);
    
//     inputs.clear();
//     targets.clear();
    
//     for (int class_id = 0; class_id < 2; class_id++) {
//         for (int i = 0; i < points_per_class; i++) {
//             double t = (double)i / points_per_class * 4.0; // 0 to 4
//             double r = t + noise(gen);
//             double angle = class_id * M_PI + t + noise(gen);
            
//             double x = r * cos(angle);
//             double y = r * sin(angle);
            
//             inputs.push_back({x, y});
//             targets.push_back({(double)class_id});
//         }
//     }
// }

// // Generate circular data
// void generate_circular_data(std::vector<std::vector<double>>& inputs, 
//                            std::vector<std::vector<double>>& targets, 
//                            int points_per_class = 100) {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<> angle_dist(0.0, 2.0 * M_PI);
//     std::normal_distribution<> noise(0.0, 0.1);
    
//     inputs.clear();
//     targets.clear();
    
//     // Inner circle (class 0)
//     for (int i = 0; i < points_per_class; i++) {
//         double angle = angle_dist(gen);
//         double r = 1.0 + noise(gen);
        
//         double x = r * cos(angle);
//         double y = r * sin(angle);
        
//         inputs.push_back({x, y});
//         targets.push_back({0});
//     }
    
//     // Outer circle (class 1)
//     for (int i = 0; i < points_per_class; i++) {
//         double angle = angle_dist(gen);
//         double r = 3.0 + noise(gen);
        
//         double x = r * cos(angle);
//         double y = r * sin(angle);
        
//         inputs.push_back({x, y});
//         targets.push_back({1});
//     }
// }

// // Shuffle data
// void shuffle_data(std::vector<std::vector<double>>& inputs, 
//                  std::vector<std::vector<double>>& targets) {
//     std::random_device rd;
//     std::mt19937 gen(rd());
    
//     std::vector<size_t> indices(inputs.size());
//     std::iota(indices.begin(), indices.end(), 0);
//     std::shuffle(indices.begin(), indices.end(), gen);
    
//     std::vector<std::vector<double>> shuffled_inputs, shuffled_targets;
//     for (size_t idx : indices) {
//         shuffled_inputs.push_back(inputs[idx]);
//         shuffled_targets.push_back(targets[idx]);
//     }
    
//     inputs = shuffled_inputs;
//     targets = shuffled_targets;
// }

// // Split data into train/test
// void split_data(const std::vector<std::vector<double>>& inputs,
//                const std::vector<std::vector<double>>& targets,
//                std::vector<std::vector<double>>& train_inputs,
//                std::vector<std::vector<double>>& train_targets,
//                std::vector<std::vector<double>>& test_inputs,
//                std::vector<std::vector<double>>& test_targets,
//                double train_ratio = 0.8) {
    
//     size_t train_size = (size_t)(inputs.size() * train_ratio);
    
//     train_inputs.clear();
//     train_targets.clear();
//     test_inputs.clear();
//     test_targets.clear();
    
//     for (size_t i = 0; i < inputs.size(); i++) {
//         if (i < train_size) {
//             train_inputs.push_back(inputs[i]);
//             train_targets.push_back(targets[i]);
//         } else {
//             test_inputs.push_back(inputs[i]);
//             test_targets.push_back(targets[i]);
//         }
//     }
// }

// int main() {
//     std::cout << "=== Binary Classification Example ===" << std::endl;
    
//     // Generate data
//     std::vector<std::vector<double>> inputs, targets;
//     std::cout << "Choose dataset:" << std::endl;
//     std::cout << "1. Spiral data" << std::endl;
//     std::cout << "2. Circular data" << std::endl;
//     std::cout << "Enter choice (1 or 2): ";
    
//     int choice;
//     std::cin >> choice;
    
//     if (choice == 1) {
//         generate_spiral_data(inputs, targets, 150);
//         std::cout << "Generated spiral dataset" << std::endl;
//     } else {
//         generate_circular_data(inputs, targets, 150);
//         std::cout << "Generated circular dataset" << std::endl;
//     }
    
//     // Shuffle data
//     shuffle_data(inputs, targets);
    
//     // Split data
//     std::vector<std::vector<double>> train_inputs, train_targets;
//     std::vector<std::vector<double>> test_inputs, test_targets;
//     split_data(inputs, targets, train_inputs, train_targets, test_inputs, test_targets);
    
//     std::cout << "Training samples: " << train_inputs.size() << std::endl;
//     std::cout << "Testing samples: " << test_inputs.size() << std::endl;
//     std::cout << std::endl;
    
//     // Create different network architectures to test
//     std::vector<std::vector<int>> architectures = {
//         {2, 8, 1},           // Simple
//         {2, 16, 8, 1},       // Deeper
//         {2, 32, 16, 1},      // Wider
//         {2, 16, 16, 8, 1}    // Even deeper
//     };
    
//     std::vector<std::string> arch_names = {
//         "Simple (2-8-1)",
//         "Deeper (2-16-8-1)",
//         "Wider (2-32-16-1)",
//         "Even Deeper (2-16-16-8-1)"
//     };
    
//     std::cout << "Choose network architecture:" << std::endl;
//     for (size_t i = 0; i < architectures.size(); i++) {
//         std::cout << i + 1 << ". " << arch_names[i] << std::endl;
//     }
//     std::cout << "Enter choice (1-" << architectures.size() << "): ";
    
//     int arch_choice;
//     std::cin >> arch_choice;
//     arch_choice = std::max(1, std::min((int)architectures.size(), arch_choice)) - 1;
    
//     std::vector<int> layer_sizes = architectures[arch_choice];
//     std::cout << "Selected architecture: " << arch_names[arch_choice] << std::endl;
    
//     // Create activation functions (ReLU for hidden layers, Sigmoid for output)
//     std::vector<ml::ActivationType> activations;
//     for (size_t i = 0; i < layer_sizes.size() - 2; i++) {
//         activations.push_back(ml::ActivationType::RELU);
//     }
//     activations.push_back(ml::ActivationType::SIGMOID);
    
//     // Create and train network
//     MLP network(layer_sizes, activations, 0.01);
//     network.print_summary();
//     std::cout << std::endl;
    
//     // Train with validation
//     std::cout << "=== Training with Validation ===" << std::endl;
    
//     // Further split training data for validation
//     std::vector<std::vector<double>> val_inputs, val_targets;
//     std::vector<std::vector<double>> final_train_inputs, final_train_targets;
//     split_data(train_inputs, train_targets, final_train_inputs, final_train_targets, val_inputs, val_targets, 0.8);
    
//     std::cout << "Final training samples: " << final_train_inputs.size() << std::endl;
//     std::cout << "Validation samples: " << val_inputs.size() << std::endl;
//     std::cout << std::endl;
    
//     double final_loss = network.train_with_validation(
//         final_train_inputs, final_train_targets,
//         val_inputs, val_targets,
//         2000, 100);
    
//     std::cout << "Final training loss: " << std::fixed << std::setprecision(6) << final_loss << std::endl;
//     std::cout << std::endl;
    
//     // Evaluate on test set
//     std::cout << "=== Final Evaluation ===" << std::endl;
    
//     double test_loss = network.evaluate(test_inputs, test_targets);
//     double test_accuracy = network.calculate_accuracy(test_inputs, test_targets);
    
//     std::cout << "Test Loss: " << std::fixed << std::setprecision(6) << test_loss << std::endl;
//     std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << test_accuracy << "%" << std::endl;
    
//     // Show some predictions
//     std::cout << "\n=== Sample Predictions ===" << std::endl;
//     for (int i = 0; i < std::min(10, (int)test_inputs.size()); i++) {
//         std::vector<double> prediction = network.predict(test_inputs[i]);
//         int predicted_class = (prediction[0] > 0.5) ? 1 : 0;
//         int actual_class = (int)test_targets[i][0];
        
//         std::cout << "Input: [" << std::fixed << std::setprecision(2) 
//                   << test_inputs[i][0] << ", " << test_inputs[i][1] << "] -> "
//                   << "Prediction: " << std::setprecision(4) << prediction[0] 
//                   << " (Class: " << predicted_class << ", Actual: " << actual_class << ")"
//                   << (predicted_class == actual_class ? " ✓" : " ✗") << std::endl;
//     }
    
//     // Save the best model
//     std::string model_name = "../models/classification_model.txt";
//     if (network.save_model(model_name)) {
//         std::cout << "\nModel saved to: " << model_name << std::endl;
//     }
    
//     return 0;
// }





#include <iostream>
#include <vector>
#include "../algorithms/mlp.hpp"

int main() {
    std::cout << "=== Simple Linear Regression Example ===" << std::endl;
    
    // Simple dataset: house size (in 1000 sqft) vs price (in $1000)
    // This represents: 1000sqft->100k, 1500sqft->150k, 2000sqft->200k, etc.
    std::vector<double> house_sizes = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    std::vector<double> prices = {100, 150, 200, 250, 300, 350, 400};
    
    std::cout << "Training data:" << std::endl;
    for (size_t i = 0; i < house_sizes.size(); i++) {
        std::cout << "Size: " << house_sizes[i] << "k sqft, Price: $" << prices[i] << "k" << std::endl;
    }
    std::cout << std::endl;
    
    // Create and train the model
    LinearRegression model(0.01);  // learning rate = 0.01
    
    std::cout << "Training model..." << std::endl;
    model.train(house_sizes, prices, 1000);
    std::cout << std::endl;
    
    // Print the trained model
    model.printModel();
    std::cout << std::endl;
    
    // Test predictions
    std::cout << "Testing predictions:" << std::endl;
    std::vector<double> test_sizes = {1.2, 2.3, 3.8};
    
    for (double size : test_sizes) {
        double predicted_price = model.predict(size);
        std::cout << "House size: " << size << "k sqft -> Predicted price: $" << predicted_price << "k" << std::endl;
    }
    
    return 0;
}