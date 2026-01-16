// src/engine.cu

#include "../include/engine.h"
#include "../include/layers.h"
#include <fstream>
#include <iostream>

InferenceEngine::InferenceEngine() : inputTensor(nullptr), outputTensor(nullptr) {
    initializeCublas();
}

InferenceEngine::~InferenceEngine() {
    cleanup();
}

void InferenceEngine::initializeCublas() {
    cublasStatus_t status = cublasCreate(&cublasHandle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        exit(1);
    }
}

void InferenceEngine::cleanup() {
    for (auto layer : layers) {
        delete layer;
    }
    layers.clear();
    
    delete inputTensor;
    delete outputTensor;
    
    cublasDestroy(cublasHandle);
}

bool InferenceEngine::loadModel(const std::string& modelPath) {
    std::ifstream file(modelPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open model file: " << modelPath << std::endl;
        return false;
    }
    
    // Read model metadata
    int layerCount;
    file.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));
    
    for (int i = 0; i < layerCount; i++) {
        int layerType;
        file.read(reinterpret_cast<char*>(&layerType), sizeof(layerType));
        
        Layer* layer = nullptr;
        
        switch (static_cast<LayerType>(layerType)) {
            case FC_LAYER: {
                int inSize, outSize;
                file.read(reinterpret_cast<char*>(&inSize), sizeof(inSize));
                file.read(reinterpret_cast<char*>(&outSize), sizeof(outSize));
                
                layer = new FCLayer(inSize, outSize);
                
                // Load weights
                std::vector<float> weights(inSize * outSize);
                file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
                
                std::vector<float> biases(outSize);
                file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
                
                // Combine weights and biases
                weights.insert(weights.end(), biases.begin(), biases.end());
                layer->loadWeights(weights.data(), weights.size());
                break;
            }
            case CONV_LAYER: {
                int inCh, outCh, kH, kW, sH, sW, pH, pW, iH, iW;
                file.read(reinterpret_cast<char*>(&inCh), sizeof(inCh));
                file.read(reinterpret_cast<char*>(&outCh), sizeof(outCh));
                file.read(reinterpret_cast<char*>(&kH), sizeof(kH));
                file.read(reinterpret_cast<char*>(&kW), sizeof(kW));
                file.read(reinterpret_cast<char*>(&sH), sizeof(sH));
                file.read(reinterpret_cast<char*>(&sW), sizeof(sW));
                file.read(reinterpret_cast<char*>(&pH), sizeof(pH));
                file.read(reinterpret_cast<char*>(&pW), sizeof(pW));
                file.read(reinterpret_cast<char*>(&iH), sizeof(iH));
                file.read(reinterpret_cast<char*>(&iW), sizeof(iW));
                
                layer = new ConvLayer(inCh, outCh, kH, kW, sH, sW, pH, pW);
                
                // Load weights
                std::vector<float> weights(inCh * outCh * kH * kW);
                file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
                
                std::vector<float> biases(outCh);
                file.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
                
                // Combine weights and biases
                weights.insert(weights.end(), biases.begin(), biases.end());
                layer->loadWeights(weights.data(), weights.size());
                break;
            }
            case ACTIVATION_LAYER: {
                int actType;
                file.read(reinterpret_cast<char*>(&actType), sizeof(actType));
                layer = new ActivationLayer(static_cast<ActivationLayer::ActivationType>(actType));
                break;
            }
            default:
                std::cerr << "Unknown layer type: " << layerType << std::endl;
                return false;
        }
        
        layers.push_back(layer);
    }
    
    file.close();
    return true;
}

void InferenceEngine::setInput(const float* inputData, int batchSize, int inputSize) {
    if (!inputTensor) {
        inputTensor = new Tensor();
    }
    
    // Resize input tensor if needed
    if (inputTensor->dims[0] != batchSize || inputTensor->dims[1] != inputSize) {
        delete inputTensor;
        inputTensor = new Tensor();
        inputTensor->allocate(batchSize, inputSize);
    }
    
    inputTensor->copyFromHost(inputData);
}

std::vector<float> InferenceEngine::runInference() {
    if (layers.empty()) {
        std::cerr << "No layers loaded" << std::endl;
        return {};
    }
    
    // Process through each layer
    Tensor* currentInput = inputTensor;
    Tensor* currentOutput = nullptr;
    
    for (size_t i = 0; i < layers.size(); i++) {
        // Allocate output tensor for this layer
        delete currentOutput;
        currentOutput = new Tensor();
        
        // This is a simplified approach - in a real implementation, 
        // you'd calculate output dimensions based on layer type
        if (i == layers.size() - 1) {
            // Last layer - assume output size of 10 for classification
            currentOutput->allocate(currentInput->dims[0], 10);
        } else {
            // For intermediate layers, just pass through the batch dimension
            currentOutput->allocate(currentInput->dims[0], currentInput->dims[1]);
        }
        
        // Run forward pass
        layers[i]->forward(currentInput, currentOutput, cublasHandle);
        
        // Swap input/output for next iteration
        currentInput = currentOutput;
    }
    
    // Copy final output to host
    output.resize(currentOutput->size);
    currentOutput->copyToHost(output.data());
    
    delete currentOutput;
    
    return output;
}