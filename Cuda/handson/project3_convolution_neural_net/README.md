# Hands-On Project 3: Convolution Neural Network Operations with CUDA

## Overview
This project implements fundamental operations for Convolutional Neural Networks (CNNs) using CUDA. You'll learn about convolution operations, activation functions, and pooling operations commonly used in deep learning.

## Learning Objectives
- Implement convolution operations on GPU
- Optimize memory access patterns for convolution
- Implement activation functions like ReLU
- Implement pooling operations (max pooling)
- Understand the computational requirements of CNN operations

## Project Structure
```
project3_convolution_neural_net/
├── README.md
├── cnn_ops.cu
├── Makefile
└── solution/
    └── cnn_ops_solution.cu
```

## Step-by-Step Guide

### Step 1: Set up the basic structure
Create a file called `cnn_ops.cu` with the basic structure:

```cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper function for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// TODO: Implement convolution kernel
__global__ void convolution(/* parameters */) {
    // Calculate position in output
    // Load input and filter data
    // Perform convolution operation
    // Handle boundary conditions
}

// TODO: Implement optimized convolution with shared memory
#define FILTER_SIZE 3
#define TILE_SIZE 16
__global__ void convolution_optimized(/* parameters */) {
    // Use shared memory to store input tile with halo
    // Load filter to registers
    // Perform convolution using shared memory
}

// TODO: Implement ReLU activation function
__global__ void relu_activation(/* parameters */) {
    // Apply ReLU: max(0, x)
}

// TODO: Implement max pooling
__global__ void max_pooling(/* parameters */) {
    // Perform max pooling operation
    // Each output element is max of corresponding input region
}

// TODO: Implement softmax activation
__global__ void softmax_activation(/* parameters */) {
    // Apply softmax: exp(x_i) / sum(exp(x_j))
    // Use shared memory for reduction
}

int main() {
    // Define network parameters
    const int INPUT_HEIGHT = 32;
    const int INPUT_WIDTH = 32;
    const int INPUT_CHANNELS = 3;
    const int FILTER_HEIGHT = 3;
    const int FILTER_WIDTH = 3;
    const int OUTPUT_CHANNELS = 16;
    
    // Calculate sizes
    const int input_size = INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS;
    const int filter_size = FILTER_HEIGHT * FILTER_WIDTH * INPUT_CHANNELS * OUTPUT_CHANNELS;
    const int output_height = INPUT_HEIGHT - FILTER_HEIGHT + 1;
    const int output_width = INPUT_WIDTH - FILTER_WIDTH + 1;
    const int output_size = output_height * output_width * OUTPUT_CHANNELS;
    
    // TODO: Allocate host memory
    float *h_input = /* allocate */;
    float *h_filter = /* allocate */;
    float *h_output = /* allocate */;
    
    // TODO: Initialize host data
    // Initialize input with random values
    // Initialize filter with random values
    
    // TODO: Allocate device memory
    float *d_input, *d_filter, *d_output;
    // cudaMalloc for each tensor
    
    // TODO: Copy data from host to device
    // cudaMemcpy for input and filter
    
    // TODO: Set up execution configuration
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x,
                  (output_height + blockSize.y - 1) / blockSize.y,
                  OUTPUT_CHANNELS);
    
    // TODO: Launch convolution kernel
    convolution<<<gridSize, blockSize>>>(d_input, d_filter, d_output,
                                         INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS,
                                         FILTER_HEIGHT, FILTER_WIDTH, OUTPUT_CHANNELS);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // TODO: Apply activation function (ReLU)
    int total_elements = output_size;
    int linear_block_size = 256;
    int linear_grid_size = (total_elements + linear_block_size - 1) / linear_block_size;
    
    relu_activation<<<linear_grid_size, linear_block_size>>>(d_output, total_elements);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // TODO: Perform pooling operation
    const int pool_size = 2;
    const int pooled_height = output_height / pool_size;
    const int pooled_width = output_width / pool_size;
    const int pooled_size = pooled_height * pooled_width * OUTPUT_CHANNELS;
    
    dim3 pool_gridSize((pooled_width + blockSize.x - 1) / blockSize.x,
                       (pooled_height + blockSize.y - 1) / blockSize.y,
                       OUTPUT_CHANNELS);
    
    max_pooling<<<pool_gridSize, blockSize>>>(d_output, d_output,  // In-place operation
                                              output_height, output_width, OUTPUT_CHANNELS,
                                              pool_size);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // TODO: Copy result back to host
    // cudaMemcpy from device to host
    
    // TODO: Print some results for verification
    printf("CNN operations completed.\n");
    printf("Input size: %dx%dx%d\n", INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS);
    printf("Output size after conv+relu+pool: %dx%dx%d\n", 
           pooled_height, pooled_width, OUTPUT_CHANNELS);
    
    // TODO: Cleanup allocated memory
    // cudaFree for device memory
    // free for host memory
    
    return 0;
}
```

### Step 2: Implement Basic Convolution
Complete the `convolution` kernel:

```cuda
__global__ void convolution(float *input, float *filter, float *output,
                           int input_h, int input_w, int input_c,
                           int filter_h, int filter_w, int output_c) {
    // Calculate output position
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch = blockIdx.z;
    
    if (out_x >= (input_w - filter_w + 1) || 
        out_y >= (input_h - filter_h + 1) || 
        out_ch >= output_c) {
        return;
    }
    
    float sum = 0.0f;
    
    // Perform convolution
    for (int f_y = 0; f_y < filter_h; f_y++) {
        for (int f_x = 0; f_x < filter_w; f_x++) {
            for (int ch = 0; ch < input_c; ch++) {
                int input_idx = ((ch * input_h + (out_y + f_y)) * input_w + (out_x + f_x));
                int filter_idx = ((((out_ch * input_c + ch) * filter_h + f_y) * filter_w + f_x));
                sum += input[input_idx] * filter[filter_idx];
            }
        }
    }
    
    int output_idx = ((out_ch * (input_h - filter_h + 1) + out_y) * (input_w - filter_w + 1) + out_x);
    output[output_idx] = sum;
}
```

### Step 3: Implement Optimized Convolution with Shared Memory
Complete the `convolution_optimized` kernel:

```cuda
#define FILTER_SIZE 3
#define TILE_SIZE 16

__global__ void convolution_optimized(float *input, float *filter, float *output,
                                     int input_h, int input_w, int input_c,
                                     int filter_h, int filter_w, int output_c) {
    // Shared memory for input tile with halo
    __shared__ float tile[TILE_SIZE + FILTER_SIZE - 1][TILE_SIZE + FILTER_SIZE - 1][INPUT_CHANNELS_PER_BLOCK];
    
    // Calculate positions
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_ch = blockIdx.z;
    
    // Load data into shared memory
    // Each thread loads multiple elements if needed
    
    __syncthreads();
    
    // Perform convolution using shared memory
    // Similar to basic convolution but using shared memory
}
```

### Step 4: Implement Activation Functions
Complete the activation functions:

```cuda
__global__ void relu_activation(float *input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        input[idx] = fmaxf(0.0f, input[idx]);
    }
}
```

### Step 5: Create a Makefile
Create a `Makefile`:

```makefile
CC = nvcc
CFLAGS = -O3 -arch=sm_50
TARGET = cnn_ops
SOURCE = cnn_ops.cu

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

.PHONY: clean
```

### Step 6: Test Your Implementation
1. Compile your code: `make`
2. Run the executable: `./cnn_ops`
3. Verify that the operations work correctly
4. Compare performance with different implementations

## Challenge Extensions
1. Implement batch processing for multiple images
2. Add support for padding in convolution
3. Implement stride in convolution
4. Add more activation functions (sigmoid, tanh)
5. Implement average pooling
6. Optimize for memory coalescing

## Solution
A complete solution is provided in the `solution/` directory for reference after you've attempted the implementation yourself.