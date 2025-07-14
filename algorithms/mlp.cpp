#include "mlp.hpp"
#include <iostream>
#include <vector>

void LinearRegression::train(const std::vector<double>& X, const std::vector<double>& y, int epochs) {
    int n = X.size();
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::vector<double> predictions = predict(X);
        double weight_gradient = 0.0, bias_gradient = 0.0;
        for (int i = 0; i < n; i++) {
            double error = predictions[i] - y[i];
            weight_gradient += error * X[i];
            bias_gradient += error;
        }
        weight_gradient /= n;
        bias_gradient /= n;
        weight -= learning_rate * weight_gradient;
        bias -= learning_rate * bias_gradient;
        if (epoch % 100 == 0) {
            double cost = calculateCost(X, y);
            std::cout << "Epoch " << epoch << ", Cost: " << cost << std::endl;
        }
    }
}

std::vector<double> LinearRegression::predict(const std::vector<double>& X) const {
    std::vector<double> predictions;
    for (double x : X) predictions.push_back(weight * x + bias);
    return predictions;
}

double LinearRegression::predict(double x) const {
    return weight * x + bias;
}

double LinearRegression::calculateCost(const std::vector<double>& X, const std::vector<double>& y) const {
    std::vector<double> predictions = predict(X);
    double cost = 0.0;
    int n = X.size();
    for (int i = 0; i < n; i++) cost += (predictions[i] - y[i]) * (predictions[i] - y[i]);
    return cost / (2 * n);
}

void LinearRegression::printModel() const {
    std::cout << "Trained model: y = " << weight << "x + " << bias << std::endl;
}