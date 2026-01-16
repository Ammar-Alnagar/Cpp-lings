// src/utils.cu

#include "../include/utils.h"
#include <fstream>
#include <iostream>

// Load binary data from file
bool loadBinaryFile(const std::string& filename, std::vector<float>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Resize vector to accommodate the data
    data.resize(fileSize / sizeof(float));
    
    // Read the data
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();
    
    return true;
}

// Save binary data to file
bool saveBinaryFile(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not create file: " << filename << std::endl;
        return false;
    }
    
    // Write the data
    file.write(reinterpret_cast<const char*>(data.data()), 
               data.size() * sizeof(float));
    file.close();
    
    return true;
}

// Print tensor information
void printTensorInfo(const float* tensor, int size, const std::string& name) {
    std::cout << "Tensor '" << name << "' info:" << std::endl;
    std::cout << "  Size: " << size << " elements" << std::endl;
    std::cout << "  First 10 elements: ";
    for (int i = 0; i < std::min(10, size); i++) {
        std::cout << tensor[i] << " ";
    }
    std::cout << std::endl;
}

// Timer utility implementation
GPUTimer::GPUTimer() {
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
}

GPUTimer::~GPUTimer() {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void GPUTimer::start() {
    cudaEventRecord(startEvent);
}

void GPUTimer::stop() {
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
}

float GPUTimer::elapsed() const {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    return milliseconds;
}