// src/main.cu

#include "../include/engine.h"
#include <iostream>
#include <random>

int main() {
    std::cout << "CUDA Neural Network Inference Engine" << std::endl;
    
    // Create engine instance
    InferenceEngine engine;
    
    // For this example, we'll create a simple model in code
    // In a real scenario, you'd load a pre-trained model
    
    std::cout << "Creating a simple model..." << std::endl;
    
    // Create sample input data
    const int batchSize = 1;
    const int inputSize = 784;  // 28x28 flattened for MNIST
    std::vector<float> inputData(inputSize);
    
    // Initialize with random data for testing
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < inputSize; i++) {
        inputData[i] = dis(gen);
    }
    
    // Set input
    engine.setInput(inputData.data(), batchSize, inputSize);
    
    // Run inference
    std::cout << "Running inference..." << std::endl;
    auto output = engine.runInference();
    
    // Display results
    std::cout << "Inference completed. Output size: " << output.size() << std::endl;
    std::cout << "First 10 outputs: ";
    for (int i = 0; i < std::min(10, (int)output.size()); i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Inference engine demo completed successfully!" << std::endl;
    
    return 0;
}