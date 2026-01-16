// tutorial/step_by_step_guide.md

# Step-by-Step Guide: Building a CUDA Neural Network Inference Engine

## Introduction
This tutorial will guide you through building a complete neural network inference engine using CUDA. By the end, you'll have a working system that can load trained model weights and perform inference on input data.

## Prerequisites
- Basic understanding of C++ and object-oriented programming
- Familiarity with neural networks and deep learning concepts
- Basic knowledge of CUDA programming (covered in previous chapters)

## Step 1: Understanding the Architecture

The inference engine follows a layered architecture:
- **Engine**: Manages the overall inference process
- **Layers**: Individual neural network operations (FC, Conv, Pool, Activation)
- **Tensors**: GPU memory management for data
- **Activations**: Non-linear activation functions

## Step 2: Setting Up the Project Structure

First, create the directory structure as outlined in the main README. This separates headers, source code, and build artifacts cleanly.

## Step 3: Implementing the Tensor Class

The Tensor class is fundamental to our engine. It manages GPU memory allocation and data transfers:

```cpp
struct Tensor {
    float* data;          // GPU memory pointer
    int dims[4];          // Dimensions [batch, channel, height, width] or [batch, features]
    int ndim;             // Number of dimensions used
    size_t size;          // Total number of elements
    size_t bytes;         // Total bytes
    
    // Constructor allocates GPU memory
    void allocate(int d0, int d1 = 1, int d2 = 1, int d3 = 1) {
        dims[0] = d0; dims[1] = d1; dims[2] = d2; dims[3] = d3;
        ndim = (d3 > 1) ? 4 : (d2 > 1) ? 3 : (d1 > 1) ? 2 : 1;
        size = d0 * d1 * d2 * d3;
        bytes = size * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&data, bytes));
    }
    
    // Copy data from host to device
    void copyFromHost(const float* hostData) {
        CUDA_CHECK(cudaMemcpy(data, hostData, bytes, cudaMemcpyHostToDevice));
    }
    
    // Copy data from device to host
    void copyToHost(float* hostData) {
        CUDA_CHECK(cudaMemcpy(hostData, data, bytes, cudaMemcpyDeviceToHost));
    }
};
```

## Step 4: Creating the Base Layer Interface

All layers inherit from a common base class:

```cpp
struct Layer {
    LayerType type;
    std::string name;
    
    Layer(LayerType t, const std::string& n) : type(t), name(n) {}
    virtual ~Layer() = default;
    
    virtual void forward(Tensor* input, Tensor* output, cublasHandle_t handle) = 0;
    virtual void loadWeights(const float* weights, size_t weightCount) = 0;
};
```

## Step 5: Implementing the Fully Connected Layer

The fully connected layer performs matrix multiplication with bias addition:

```cpp
struct FCLayer : public Layer {
    Tensor weights;      // [input_size, output_size]
    Tensor biases;       // [output_size]
    int inputSize;
    int outputSize;
    
    // Implementation of forward pass using cuBLAS
    void forward(Tensor* input, Tensor* output, cublasHandle_t handle) {
        // Matrix multiplication: output = input * weights + biases
        const float alpha = 1.0f, beta = 1.0f;
        int batchSize = input->dims[0];
        
        for (int b = 0; b < batchSize; b++) {
            cublasSgemv(handle, CUBLAS_OP_N,
                        outputSize, inputSize,
                        &alpha,
                        weights.data + b * inputSize * outputSize, outputSize,
                        input->data + b * inputSize, 1,
                        &beta,
                        output->data + b * outputSize, 1);
        }
    }
};
```

## Step 6: Adding Activation Functions

Activation functions introduce non-linearity to the network:

```cpp
__global__ void reluKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}
```

## Step 7: Building and Testing

Use the provided Makefile to build the project:

```bash
make
./bin/inference_engine
```

## Step 8: Extending the Engine

Consider adding these features to enhance your engine:
- Support for different data types (FP16, INT8)
- More layer types (BatchNorm, Dropout)
- Model quantization support
- Multi-GPU inference
- ONNX model loading

## Troubleshooting Common Issues

1. **Memory allocation failures**: Check GPU memory availability
2. **Kernel launch errors**: Verify grid/block dimensions
3. **Wrong results**: Check weight ordering and dimensions
4. **Performance issues**: Profile with Nsight and optimize memory access

## Conclusion

You've now built a functional CUDA-based inference engine! This serves as a foundation that can be extended with more sophisticated features and optimizations.