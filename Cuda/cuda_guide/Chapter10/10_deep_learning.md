# Chapter 10: Deep Learning with CUDA

## Overview
Deep learning has become one of the most important applications of GPU computing. CUDA provides the foundation for many deep learning frameworks and enables efficient training and inference of neural networks. This chapter covers the fundamentals of deep learning on GPUs, CUDA-specific optimizations, and how to implement basic neural network components.

## Neural Network Fundamentals

### Basic Neural Network Components
Neural networks consist of layers of interconnected neurons. The key operations that benefit from GPU acceleration are:

1. **Matrix multiplications** (fully connected layers)
2. **Convolution operations** (convolutional layers)
3. **Activation functions** (ReLU, sigmoid, etc.)
4. **Pooling operations** (max pooling, average pooling)
5. **Loss function calculations**
6. **Backpropagation gradients**

### Mathematical Operations in Neural Networks
```cuda
// Matrix multiplication for fully connected layer
__global__ void matrixMul(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Activation function: ReLU
__global__ void reluActivation(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Activation function: Sigmoid
__global__ void sigmoidActivation(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}
```

## CUDA Optimizations for Deep Learning

### Memory Layout Optimizations
```cuda
// Optimized memory access for batch processing
__global__ void batchMatrixMul(const float *A, const float *B, float *C,
                              int batchSize, int M, int N, int K) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < batchSize && row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[batch * M * K + row * K + k] * 
                   B[batch * K * N + k * N + col];
        }
        C[batch * M * N + row * N + col] = sum;
    }
}
```

### Shared Memory for Convolution
```cuda
#define TILE_SIZE 16
#define FILTER_SIZE 3
#define FILTER_RADIUS (FILTER_SIZE/2)

__global__ void convolutionOptimized(float *input, float *output, float *filter,
                                   int width, int height, int channels) {
    __shared__ float tile[TILE_SIZE + 2*FILTER_RADIUS][TILE_SIZE + 2*FILTER_RADIUS];
    
    int channel = blockIdx.z;
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load data into shared memory with halo
    for (int dy = 0; dy < TILE_SIZE + 2*FILTER_RADIUS; dy += TILE_SIZE) {
        for (int dx = 0; dx < TILE_SIZE + 2*FILTER_RADIUS; dx += TILE_SIZE) {
            int src_y = y + dy - FILTER_RADIUS;
            int src_x = x + dx - FILTER_RADIUS;
            
            if (src_y >= 0 && src_y < height && src_x >= 0 && src_x < width) {
                tile[threadIdx.y + dy][threadIdx.x + dx] = 
                    input[channel * height * width + src_y * width + src_x];
            } else {
                tile[threadIdx.y + dy][threadIdx.x + dx] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // Perform convolution
    if (y < height && x < width) {
        float sum = 0.0f;
        for (int fy = 0; fy < FILTER_SIZE; fy++) {
            for (int fx = 0; fx < FILTER_SIZE; fx++) {
                sum += tile[threadIdx.y + fy][threadIdx.x + fx] * 
                       filter[fy * FILTER_SIZE + fx];
            }
        }
        output[channel * height * width + y * width + x] = sum;
    }
}
```

## cuDNN Library for Deep Learning

### Using cuDNN for Convolution
```cpp
#include <cudnn.h>
#include <cuda_runtime.h>

class CudnnConvolution {
private:
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t srcDesc, dstDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnActivationDescriptor_t activationDesc;

public:
    CudnnConvolution() {
        cudnnCreate(&handle);
        cudnnCreateTensorDescriptor(&srcDesc);
        cudnnCreateTensorDescriptor(&dstDesc);
        cudnnCreateFilterDescriptor(&filterDesc);
        cudnnCreateConvolutionDescriptor(&convDesc);
        cudnnCreateActivationDescriptor(&activationDesc);
    }
    
    void forward(float *input, float *filter, float *output,
                 int batchSize, int channels, int height, int width,
                 int filterHeight, int filterWidth) {
        // Set tensor descriptors
        cudnnSetTensor4dDescriptor(srcDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   batchSize, channels, height, width);
        cudnnSetTensor4dDescriptor(dstDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   batchSize, channels, height, width);
        cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                   channels, channels, filterHeight, filterWidth);
        cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1,
                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        
        // Find best algorithm
        cudnnConvolutionFwdAlgo_t algo;
        cudnnGetConvolutionForwardAlgorithm(handle, srcDesc, filterDesc, 
                                          convDesc, dstDesc,
                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                          0, &algo);
        
        // Get workspace size
        size_t workspaceSize;
        cudnnGetConvolutionForwardWorkspaceSize(handle, srcDesc, filterDesc,
                                               convDesc, dstDesc, algo,
                                               &workspaceSize);
        
        // Allocate workspace
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            cudaMalloc(&workspace, workspaceSize);
        }
        
        // Perform convolution
        float alpha = 1.0f, beta = 0.0f;
        cudnnConvolutionForward(handle, &alpha, srcDesc, input,
                               filterDesc, filter,
                               convDesc, algo, workspace, workspaceSize,
                               &beta, dstDesc, output);
        
        // Cleanup
        if (workspace) cudaFree(workspace);
    }
    
    ~CudnnConvolution() {
        cudnnDestroy(handle);
        cudnnDestroyTensorDescriptor(srcDesc);
        cudnnDestroyTensorDescriptor(dstDesc);
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyActivationDescriptor(activationDesc);
    }
};
```

## Basic Neural Network Layer Implementation

### Fully Connected Layer
```cuda
class CudaFullyConnectedLayer {
private:
    float *weights, *bias, *output;
    int inputSize, outputSize;
    
public:
    CudaFullyConnectedLayer(int inputSz, int outputSz) 
        : inputSize(inputSz), outputSize(outputSz) {
        // Allocate GPU memory
        cudaMalloc(&weights, inputSize * outputSize * sizeof(float));
        cudaMalloc(&bias, outputSize * sizeof(float));
        cudaMalloc(&output, outputSize * sizeof(float));
        
        // Initialize weights randomly
        initializeWeights();
    }
    
    void forward(float *input) {
        // Perform matrix multiplication: output = input * weights + bias
        const float alpha = 1.0f, beta = 1.0f;
        
        // Use cuBLAS for optimized matrix multiplication
        cublasHandle_t cublasH;
        cublasCreate(&cublasH);
        
        // Initialize output with bias
        cudaMemcpy(output, bias, outputSize * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // Perform matrix multiplication: output = input * weights + output
        cublasSgemv(cublasH, CUBLAS_OP_T, 
                    outputSize, inputSize,
                    &alpha, weights, outputSize,
                    input, 1,
                    &beta, output, 1);
        
        cublasDestroy(cublasH);
    }
    
    void initializeWeights() {
        // Initialize weights with small random values
        float *h_weights = (float*)malloc(inputSize * outputSize * sizeof(float));
        float *h_bias = (float*)malloc(outputSize * sizeof(float));
        
        for (int i = 0; i < inputSize * outputSize; i++) {
            h_weights[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        }
        for (int i = 0; i < outputSize; i++) {
            h_bias[i] = 0.0f;
        }
        
        cudaMemcpy(weights, h_weights, inputSize * outputSize * sizeof(float), 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(bias, h_bias, outputSize * sizeof(float), 
                   cudaMemcpyHostToDevice);
        
        free(h_weights);
        free(h_bias);
    }
    
    float* getOutput() { return output; }
    
    ~CudaFullyConnectedLayer() {
        cudaFree(weights);
        cudaFree(bias);
        cudaFree(output);
    }
};
```

### Batch Normalization Layer
```cuda
__global__ void batchNormForward(float *input, float *output, 
                                float *scale, float *bias,
                                float *runningMean, float *runningVar,
                                int n, int c, int h, int w, 
                                float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = n * c * h * w;
    
    if (idx < totalElements) {
        int c_idx = (idx / (h * w)) % c;  // Channel index
        
        float normalized = (input[idx] - runningMean[c_idx]) / 
                          sqrtf(runningVar[c_idx] + epsilon);
        output[idx] = normalized * scale[c_idx] + bias[c_idx];
    }
}

class CudaBatchNormLayer {
private:
    float *scale, *bias, *runningMean, *runningVar;
    int channels;
    float momentum, epsilon;
    
public:
    CudaBatchNormLayer(int ch, float mom = 0.1f, float eps = 1e-5f)
        : channels(ch), momentum(mom), epsilon(eps) {
        cudaMalloc(&scale, channels * sizeof(float));
        cudaMalloc(&bias, channels * sizeof(float));
        cudaMalloc(&runningMean, channels * sizeof(float));
        cudaMalloc(&runningVar, channels * sizeof(float));
        
        initializeParams();
    }
    
    void forward(float *input, float *output, int n, int h, int w) {
        int totalElements = n * channels * h * w;
        int blockSize = 256;
        int gridSize = (totalElements + blockSize - 1) / blockSize;
        
        batchNormForward<<<gridSize, blockSize>>>(
            input, output, scale, bias, runningMean, runningVar,
            n, channels, h, w, epsilon);
    }
    
    void initializeParams() {
        float *h_scale = (float*)malloc(channels * sizeof(float));
        float *h_bias = (float*)malloc(channels * sizeof(float));
        float *h_mean = (float*)malloc(channels * sizeof(float));
        float *h_var = (float*)malloc(channels * sizeof(float));
        
        for (int i = 0; i < channels; i++) {
            h_scale[i] = 1.0f;      // Scale starts at 1
            h_bias[i] = 0.0f;       // Bias starts at 0
            h_mean[i] = 0.0f;       // Running mean starts at 0
            h_var[i] = 1.0f;        // Running variance starts at 1
        }
        
        cudaMemcpy(scale, h_scale, channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(bias, h_bias, channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(runningMean, h_mean, channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(runningVar, h_var, channels * sizeof(float), cudaMemcpyHostToDevice);
        
        free(h_scale);
        free(h_bias);
        free(h_mean);
        free(h_var);
    }
    
    ~CudaBatchNormLayer() {
        cudaFree(scale);
        cudaFree(bias);
        cudaFree(runningMean);
        cudaFree(runningVar);
    }
};
```

## Optimized Operations for Deep Learning

### Tensor Core Operations for Deep Learning
```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <wmma.h>

using namespace nvcuda;

// Using Tensor Cores for mixed precision training
__global__ void tensorCoreGemm(half *a, half *b, float *c, int n) {
    // Tile using Tensor Cores (16x16x16 matrices)
    const int BLOCK_M = 16;
    const int BLOCK_N = 16;
    const int BLOCK_K = 16;
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, BLOCK_M, BLOCK_N, BLOCK_K, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, BLOCK_M, BLOCK_N, BLOCK_K, float> frag_c;
    
    // Calculate thread's tile
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    if (warpM * BLOCK_M >= n || warpN * BLOCK_N >= n) return;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(frag_c, 0.0f);
    
    // Load matrices and perform GEMM
    wmma::load_matrix_sync(frag_a, a + (warpM * BLOCK_M * n), n);
    wmma::load_matrix_sync(frag_b, b + (warpN * BLOCK_N), n, wmma::MEM_ROW_MAJOR);
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    wmma::store_matrix_sync(c + (warpM * BLOCK_M * n + warpN * BLOCK_N), frag_c, 
                           n, wmma::MEM_ROW_MAJOR);
}
```

### Optimized Activation Functions
```cuda
// Swish activation: x * sigmoid(x)
__global__ void swishActivation(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid_x;
    }
}

// Leaky ReLU activation
__global__ void leakyReluActivation(float *input, float *output, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float x = input[idx];
        output[idx] = (x > 0) ? x : alpha * x;
    }
}

// Softmax activation
__global__ void softmaxActivation(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Load data to shared memory
    if (bid * blockDim.x + tid < n) {
        sdata[tid] = input[bid * blockDim.x + tid];
    } else {
        sdata[tid] = -INFINITY;
    }
    __syncthreads();
    
    // Find maximum value
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (bid * blockDim.x + tid + s) < n) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    float max_val = sdata[0];
    __syncthreads();
    
    // Compute exponentials and sum
    if (bid * blockDim.x + tid < n) {
        sdata[tid] = expf(input[bid * blockDim.x + tid] - max_val);
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Sum the exponentials
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float sum_exp = sdata[0];
    __syncthreads();
    
    // Compute softmax
    if (bid * blockDim.x + tid < n) {
        output[bid * blockDim.x + tid] = sdata[tid] / sum_exp;
    }
}
```

## Memory Management for Deep Learning

### Unified Memory for Deep Learning
```cuda
class UnifiedMemoryNeuralNet {
private:
    float *input, *weights, *output;
    int inputSize, outputSize;
    
public:
    UnifiedMemoryNeuralNet(int inpSz, int outSz) 
        : inputSize(inpSz), outputSize(outSz) {
        // Allocate unified memory
        cudaMallocManaged(&input, inputSize * sizeof(float));
        cudaMallocManaged(&weights, inputSize * outputSize * sizeof(float));
        cudaMallocManaged(&output, outputSize * sizeof(float));
        
        // Initialize on CPU
        initializeOnCPU();
    }
    
    void initializeOnCPU() {
        // Initialize weights with small random values
        for (int i = 0; i < inputSize * outputSize; i++) {
            weights[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        }
        
        // Prefetch to GPU for processing
        cudaMemPrefetchAsync(weights, inputSize * outputSize * sizeof(float), 0);
    }
    
    void forward() {
        // Process on GPU
        int blockSize = 256;
        int gridSize = (outputSize + blockSize - 1) / blockSize;
        
        fullyConnectedKernel<<<gridSize, blockSize>>>(
            input, weights, output, inputSize, outputSize);
        
        cudaDeviceSynchronize();
    }
    
    void setInput(float *inp) {
        // Copy input to unified memory
        for (int i = 0; i < inputSize; i++) {
            input[i] = inp[i];
        }
    }
    
    float* getOutput() { return output; }
    
    ~UnifiedMemoryNeuralNet() {
        cudaFree(input);
        cudaFree(weights);
        cudaFree(output);
    }
};

__global__ void fullyConnectedKernel(float *input, float *weights, float *output,
                                    int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            sum += input[i] * weights[idx * inputSize + i];
        }
        output[idx] = sum;
    }
}
```

## Performance Optimization Techniques

### Memory Coalescing for Deep Learning
```cuda
// Optimized matrix multiplication with coalesced memory access
__global__ void coalescedMatrixMul(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    // Use shared memory tiles for coalesced access
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + 15) / 16; t++) {
        // Load tiles with coalesced access
        if (row < M && (t * 16 + tx) < K) {
            As[ty][tx] = A[row * K + t * 16 + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * 16 + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * 16 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < 16; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Streaming for Large Models
```cuda
class StreamedNeuralNetwork {
private:
    cudaStream_t *streams;
    float **d_input, **d_output;
    int numStreams;
    int chunkSize;
    
public:
    StreamedNeuralNetwork(int nStreams, int totalSize) 
        : numStreams(nStreams) {
        streams = new cudaStream_t[numStreams];
        d_input = new float*[numStreams];
        d_output = new float*[numStreams];
        
        chunkSize = totalSize / numStreams;
        
        // Create streams and allocate memory
        for (int i = 0; i < numStreams; i++) {
            cudaStreamCreate(&streams[i]);
            cudaMalloc(&d_input[i], chunkSize * sizeof(float));
            cudaMalloc(&d_output[i], chunkSize * sizeof(float));
        }
    }
    
    void processInChunks(float *h_input, float *h_output, int totalSize) {
        for (int i = 0; i < numStreams; i++) {
            int offset = i * chunkSize;
            int currentChunkSize = min(chunkSize, totalSize - offset);
            
            // Async memory copy
            cudaMemcpyAsync(d_input[i], h_input + offset, 
                           currentChunkSize * sizeof(float),
                           cudaMemcpyHostToDevice, streams[i]);
            
            // Launch kernel asynchronously
            processKernel<<<(currentChunkSize + 255) / 256, 256, 0, streams[i]>>>(
                d_input[i], d_output[i], currentChunkSize);
            
            // Copy result back asynchronously
            cudaMemcpyAsync(h_output + offset, d_output[i], 
                           currentChunkSize * sizeof(float),
                           cudaMemcpyDeviceToHost, streams[i]);
        }
        
        // Synchronize all streams
        for (int i = 0; i < numStreams; i++) {
            cudaStreamSynchronize(streams[i]);
        }
    }
    
    ~StreamedNeuralNetwork() {
        for (int i = 0; i < numStreams; i++) {
            cudaStreamDestroy(streams[i]);
            cudaFree(d_input[i]);
            cudaFree(d_output[i]);
        }
        delete[] streams;
        delete[] d_input;
        delete[] d_output;
    }
};

__global__ void processKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Example processing: apply activation function
        float x = input[idx];
        output[idx] = x > 0 ? x : 0.01f * x;  // Leaky ReLU
    }
}
```

## Framework Integration

### Custom CUDA Kernels for PyTorch
```cpp
// Example of a custom CUDA kernel that could be integrated with PyTorch
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void custom_activation_kernel(
    const float* input,
    float* output,
    const float alpha,
    const int64_t n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x * tanhf(logf(1.0f + expf(fminf(x, 20.0f))) * alpha);
    }
}

// Host function
torch::Tensor custom_activation(torch::Tensor input, float alpha) {
    auto output = torch::zeros_like(input);
    
    const int64_t n = input.numel();
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;
    
    custom_activation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        alpha,
        n
    );
    
    return output;
}

// Binding for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_activation", &custom_activation, "Custom activation function (CUDA)");
}
```

## Summary
This chapter covered:
- Neural network fundamentals and mathematical operations
- CUDA optimizations for deep learning operations
- Using cuDNN library for optimized deep learning primitives
- Implementation of basic neural network layers in CUDA
- Tensor Core operations for mixed precision training
- Optimized activation functions
- Memory management techniques for deep learning
- Performance optimization strategies
- Streaming for large models
- Integration with deep learning frameworks

CUDA provides the computational foundation for modern deep learning, enabling efficient training and inference of neural networks. The combination of specialized libraries like cuDNN and direct CUDA programming allows for maximum performance in deep learning applications.