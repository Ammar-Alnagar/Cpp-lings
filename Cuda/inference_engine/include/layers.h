// include/layers.h

#ifndef LAYERS_H
#define LAYERS_H

#include "engine.h"

// Fully Connected (Dense) Layer
struct FCLayer : public Layer {
    Tensor weights;      // [input_size, output_size]
    Tensor biases;       // [output_size]
    int inputSize;
    int outputSize;

    FCLayer(int inSize, int outSize);
    ~FCLayer() = default;

    void forward(Tensor* input, Tensor* output, cublasHandle_t handle) override;
    void loadWeights(const float* weights, size_t weightCount) override;
};

// Convolution Layer
struct ConvLayer : public Layer {
    Tensor weights;      // [output_channels, input_channels, kernel_h, kernel_w]
    Tensor biases;       // [output_channels]
    int inputChannels;
    int outputChannels;
    int kernelH, kernelW;
    int strideH, strideW;
    int padH, padW;
    int inputH, inputW;
    int outputH, outputW;

    ConvLayer(int inCh, int outCh, int kH, int kW, int sH = 1, int sW = 1, int pH = 0, int pW = 0);
    ~ConvLayer() = default;

    void forward(Tensor* input, Tensor* output, cublasHandle_t handle) override;
    void loadWeights(const float* weights, size_t weightCount) override;

private:
    void im2col(const float* input, float* output, int batchSize);
};

// Pooling Layer
struct PoolLayer : public Layer {
    int kernelH, kernelW;
    int strideH, strideW;
    int inputH, inputW, channels;
    int outputH, outputW;

    PoolLayer(int kH, int kW, int sH = 1, int sW = 1);
    ~PoolLayer() = default;

    void forward(Tensor* input, Tensor* output, cublasHandle_t handle) override;
    void loadWeights(const float* weights, size_t weightCount) override;
};

#endif // LAYERS_H