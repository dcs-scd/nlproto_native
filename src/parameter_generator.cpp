#include "benchmark_types.hpp"
#include <cmath>
#include <stdexcept>
#include <sstream>

std::vector<double> ParameterConfiguration::generate() const {
    switch (type) {
        case RANGE:
            return generate_range();
        case LIST:
            return values;
        case LOGARITHMIC:
            return generate_logarithmic();
        case CUSTOM:
            return generate_custom();
        default:
            throw std::runtime_error("Unknown parameter specification type");
    }
}

std::vector<double> ParameterConfiguration::generate_range() const {
    std::vector<double> result;
    
    if (step <= 0) {
        throw std::runtime_error("Step must be positive for range parameter");
    }
    
    if (min > max) {
        throw std::runtime_error("Min must be <= max for range parameter");
    }
    
    for (double val = min; val <= max + 1e-10; val += step) {  // Small epsilon for floating point comparison
        result.push_back(val);
    }
    
    return result;
}

std::vector<double> ParameterConfiguration::generate_logarithmic() const {
    std::vector<double> result;
    
    if (n <= 0) {
        throw std::runtime_error("n must be positive for logarithmic parameter");
    }
    
    if (base <= 0 || base == 1) {
        throw std::runtime_error("Base must be positive and != 1 for logarithmic parameter");
    }
    
    // Generate n logarithmic values
    // For base 10: 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1
    // For base 2: 1, 2, 4, 8, 16
    
    double start_exp = -(n-1);  // Start from negative exponents for base 10
    if (base == 2.0) {
        start_exp = 0;  // Start from 0 for base 2 (powers: 2^0, 2^1, 2^2, ...)
    }
    
    for (int i = 0; i < n; ++i) {
        double exponent = start_exp + i;
        double value = std::pow(base, exponent);
        result.push_back(value);
    }
    
    return result;
}

std::vector<double> ParameterConfiguration::generate_custom() const {
    // For now, custom is the same as list
    // This can be extended for more complex custom specifications
    return values;
}