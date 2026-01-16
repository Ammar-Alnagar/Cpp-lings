// include/utils.h

#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <vector>
#include <string>

// Utility functions for the inference engine

// Load binary data from file
bool loadBinaryFile(const std::string& filename, std::vector<float>& data);

// Save binary data to file
bool saveBinaryFile(const std::string& filename, const std::vector<float>& data);

// Print tensor information
void printTensorInfo(const float* tensor, int size, const std::string& name);

// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Timer utility
class GPUTimer {
public:
    GPUTimer();
    ~GPUTimer();
    
    void start();
    void stop();
    float elapsed() const;  // Returns time in milliseconds
    
private:
    cudaEvent_t startEvent, stopEvent;
};

#endif // UTILS_H