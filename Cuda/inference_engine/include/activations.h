// include/activations.h

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "engine.h"
#include "layers.h"

// Activation layer
struct ActivationLayer : public Layer {
    enum ActivationType {
        RELU,
        SIGMOID,
        TANH
    };

    ActivationType activationType;

    ActivationLayer(ActivationType actType);
    ~ActivationLayer() = default;

    void forward(Tensor* input, Tensor* output, cublasHandle_t handle) override;
    void loadWeights(const float* weights, size_t weightCount) override;
};

// CUDA kernels for activations
__global__ void reluKernel(float* input, float* output, int n);
__global__ void sigmoidKernel(float* input, float* output, int n);
__global__ void tanhKernel(float* input, float* output, int n);

#endif // ACTIVATIONS_H