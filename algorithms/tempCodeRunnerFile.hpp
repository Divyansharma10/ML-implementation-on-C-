ifndef LINEAR_REGRESSION_H
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