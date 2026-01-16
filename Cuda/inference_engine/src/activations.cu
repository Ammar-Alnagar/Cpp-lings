// src/activations.cu

#include "../include/activations.h"

ActivationLayer::ActivationLayer(ActivationType actType)
    : Layer(ACTIVATION_LAYER, "Activation"), activationType(actType) {}

void ActivationLayer::forward(Tensor* input, Tensor* output, cublasHandle_t handle) {
    // Output tensor should have same dimensions as input
    if (output->size != input->size) {
        delete output;
        output = new Tensor();
        output->allocate(input->dims[0], input->dims[1], input->dims[2], input->dims[3]);
    }
    
    int n = input->size;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    switch (activationType) {
        case RELU:
            reluKernel<<<gridSize, blockSize>>>(input->data, output->data, n);
            break;
        case SIGMOID:
            sigmoidKernel<<<gridSize, blockSize>>>(input->data, output->data, n);
            break;
        case TANH:
            tanhKernel<<<gridSize, blockSize>>>(input->data, output->data, n);
            break;
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ActivationLayer::loadWeights(const float* weights, size_t weightCount) {
    // Activation layers don't have weights
}

// CUDA kernels for activations
__global__ void reluKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void sigmoidKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

__global__ void tanhKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}